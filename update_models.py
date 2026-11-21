import json
import re
import requests
import logging
from pathlib import Path

# --- Configuration: All filtering rules are defined here for easy tuning ---

# The URL for fetching the list of all models from the OpenRouter API.
# NOTE: xAI models come from this same catalogue (provider "x-ai") —
# no separate x.ai API polling is needed anymore.
API_URL = "https://openrouter.ai/api/v1/models"

# Define project root and paths. In this new repo, the script is at the root.
PROJECT_ROOT = Path(__file__).parent
OUTPUT_FILE = PROJECT_ROOT / "models.json"

# --- Step 1: Technical Filters ---
# Models must ACCEPT text AND images (text for chat/terminal/browser actions,
# images so vision works), and must OUTPUT TEXT ONLY. Pure media-generation
# models (image/video/audio, e.g. gemini-3-pro-image "Nano Banana") are useless
# to us: every mode consumes text. So we reject any model whose OUTPUT
# modalities include image/video/audio.
REQUIRED_INPUT_MODALITIES = {"text", "image"}
FORBIDDEN_OUTPUT_MODALITIES = {"image", "video", "audio"}

# --- Step 2: Name-based Filters (Regular Expressions) ---
# These patterns help exclude temporary, preview, or specialized models.
# Pattern to detect dates like '20241022', '2024-11-20', or '09-2025'.
DATE_PATTERN = re.compile(r'\d{8}|\d{4}-\d{2}-\d{2}|\d{4}-\d{2}')
# Pattern to detect words indicating a non-production or test version.
PREVIEW_PATTERN = re.compile(r'\b(preview|beta|test|dev|alpha|instruct)\b', re.IGNORECASE)
# Pattern to detect models specialized for tasks other than a general assistant.
SPECIALIZED_PATTERN = re.compile(r'\b(codex|code|sql|translate|thinking)\b', re.IGNORECASE)

# --- Step 3: Quality & Capability Filters ---
# NOTE: the old provider whitelist is gone on purpose. A year ago only ~5
# providers had models capable of vision + structured JSON output; now many do
# (Qwen, Zhipu, Kimi, DeepSeek, ByteDance, Xiaomi, Mistral, Llama, ...).
# Selection is by *capability* (multimodal + tools/tool_choice + context),
# not by brand. Direct API lists stay limited to our 4 native providers;
# everything else lands in OpenRouter groups.
# Blocklist for catalogue entries that are routers/aliases, not real models.
BLOCKED_PROVIDERS = {'openrouter', 'typesafe'}
# Minimum context length. Models below this are dropped: the agent needs a
# large window to hold tool schemas + history, so we require at least 200k.
MIN_CONTEXT_LENGTH = 200000
# The model MUST support these parameters to be controllable by our
# application for structured output and tool use.
REQUIRED_PARAMETERS = {
    "tool_choice",
    "tools",
}
# The model MUST also expose a reasoning capability (any of these parameter
# names counts). Reasoning is required so the agent can plan multi-step
# terminal/browser actions.
REQUIRED_REASONING_PARAMS = {
    "reasoning",
    "include_reasoning",
    "reasoning_effort",
}

# --- Step 4: Manual Overrides ---
# This allows us to manually correct any mistakes made by the automated filters.
# Models in this list will be added to the final list, even if they were filtered out.
# Example of how to use:
# FORCE_INCLUDE = set([
#     "x-ai/grok-4-fast:free", # User confirmed this model works well.
# ])
FORCE_INCLUDE = set()
# Models in this list will be removed from the final list, even if they passed all filters.
FORCE_EXCLUDE = {
    # Reasoning model returns empty responses in practice (no usable
    # text/tool_calls in agent loops) — verified in live testing, Oct 2026.
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
}

# --- Static fallbacks: known-good models per direct provider. ---
# The OpenRouter catalogue changes constantly (ids get renamed/removed, e.g.
# anthropic/claude-sonnet-4-5 and x-ai/grok-4-fast are currently absent), so
# the dynamic result is ALWAYS merged with these. Guarantees the output never
# collapses to a single junk entry again.
STATIC_DIRECT_MODELS = {
    "google": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    "openai": ["gpt-5-mini", "gpt-5", "gpt-5-nano"],
    "anthropic": ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"],
    "x-ai": ["grok-4-fast-reasoning", "grok-4-1-fast-reasoning", "grok-4"],
}
STATIC_OPENROUTER_GROUPS = {
    "Anthropic": ["claude-sonnet-4-5", "claude-haiku-4-5"],
    "Google": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "OpenAI": ["gpt-5-mini", "gpt-5"],
    "X-AI": ["grok-4-1-fast-reasoning"],
    "Meta": ["muse-glimmer-30b"],
}
# Hugging Face Router models are NOT in the OpenRouter catalogue, so they are
# curated statically. Suffix ":provider" pins the inference provider;
# ids without suffix are auto-routed by HF.
STATIC_HUGGINGFACE_MODELS = [
    "Qwen/Qwen3.8-27B:novita",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "openai/gpt-oss-120b",
    "zai-org/GLM-4.5V",
    "meta-llama/Llama-3.1-8B-Instruct",
]

# --- Step 5: Per-mode capabilities (consumed by the Logos app) ---
# The app does not decide what a model can do; it just reads this block from
# models.json. There are only two groups:
#   general - Chat, Terminal and Browser share the SAME requirements
#             (multimodal input + tools/tool_choice, already enforced by the
#             filters), so a model that fits one fits all. Keeping them as a
#             single list prevents the requirements from drifting apart and
#             producing a chat-only model that can't run terminal/browser.
#   vision  - native computer use only, curated per vendor below.
# Keys are "provider::model" (matching what Logos stores in settings.json;
# OpenRouter ids are "group/model").
CUA_PROVIDERS = {'openai', 'anthropic', 'google'}
CUA_MODEL_PATTERNS = {
    'anthropic': r'^claude-',                                  # all current Claude models
    'openai': r'(computer-use|gpt-5\.[6-9]|gpt-[6-9])',        # Responses "computer" tool
    # Gemini 2.5+/3+, excluding lite / image-generation / nano-banana variants.
    'google': r'^gemini-(?:2\.5|[3-9])(?![^\s]*?(?:lite|image|nano-banana))',
}


def _is_vision_capable(provider_id: str, model_name: str) -> bool:
    """Whether a directly-served model exposes native computer use."""
    pattern = CUA_MODEL_PATTERNS.get(provider_id)
    return bool(pattern and re.search(pattern, model_name.lower()))

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_direct_api_model_name(model):
    """
    Intelligently determines the best model name for direct API calls. For
    Anthropic, it uses the 'id' field and replaces dots with hyphens. For
    others, it compares 'id' and 'canonical_slug' to find the cleanest,
    most stable alias.
    """
    model_id = model.get('id', '')
    provider = model_id.split('/')[0]
    base_id_name = model_id.split('/')[-1]

    # --- Special Handling for Anthropic ---
    # For Anthropic, we use the model ID directly, replacing dots with hyphens.
    if provider == 'anthropic':
        return base_id_name.replace('.', '-')

    # --- Standard Logic for Other Providers ---
    # Use the model_id as a fallback if canonical_slug is missing or empty
    canonical_slug = model.get('canonical_slug') or model_id
    base_slug_name = canonical_slug.split('/')[-1]

    # Create cleaned versions by removing all digits, hyphens, and dots.
    cleaned_id = re.sub(r'[-\d.]', '', base_id_name)
    cleaned_slug = re.sub(r'[-\d.]', '', base_slug_name)

    # If the core, non-numeric parts are the same, it's safe to assume
    # the 'id' is the desired alias.
    if cleaned_id == cleaned_slug:
        return base_id_name

    # Otherwise, they are fundamentally different. Be safe and return the
    # specific version from the slug.
    return base_slug_name

def get_modalities(model):
    """Returns (input_modalities, output_modalities) as sets of strings.

    Prefers the explicit architecture.input_modalities / output_modalities
    lists; falls back to parsing architecture.modality ("in->out")."""
    arch = model.get('architecture', {}) or {}
    modality = arch.get('modality', '') or ''

    inputs = arch.get('input_modalities')
    if not isinstance(inputs, list) or not inputs:
        left = modality.split('->')[0]
        inputs = [m.strip() for m in left.split('+') if m.strip()]

    outputs = arch.get('output_modalities')
    if not isinstance(outputs, list) or not outputs:
        right = modality.split('->')[-1]
        outputs = [m.strip() for m in right.split('+') if m.strip()]

    return set(inputs), set(outputs)


# Canonical ordering of effort levels used by the app's model picker. Values
# the catalogue reports but this list does not know yet are kept after the
# known ones, so a new level is not silently dropped.
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def get_reasoning_efforts(model):
    """Supported reasoning efforts and the default one.

    Returns ``(None, "")`` when the catalogue does not describe them for this
    model (the app then hides the effort control). The default is returned
    only when it is one of the supported levels.
    """
    reasoning = model.get('reasoning') or {}
    raw = reasoning.get('supported_efforts')
    if not isinstance(raw, list):
        return None, ""

    efforts = []
    for value in raw:
        effort = str(value or "").strip().lower()
        if effort and effort not in efforts:
            efforts.append(effort)
    if not efforts:
        return None, ""

    rank = {name: index for index, name in enumerate(EFFORT_ORDER)}
    efforts.sort(key=lambda name: (rank.get(name, len(rank)), name))

    default = str(reasoning.get('default_effort') or "").strip().lower()
    return efforts, default if default in efforts else ""


def get_model_metadata(model):
    """
    Extracts the fields Logos shows in its model info panel (and uses for
    context-overflow detection) from a raw OpenRouter catalogue entry:
      context_window    - max input tokens (model['context_length'])
      reasoning         - whether the model exposes a reasoning capability
      reasoning_efforts - effort levels accepted for reasoning, in canonical
                          order (present only when the catalogue reports them)
      default_effort    - catalogue default level (present only when valid)
      input_modalities  - e.g. ['text', 'image'] (falls back to parsing
                          architecture.modality, e.g. 'text+image->text')
    """
    arch = model.get('architecture', {}) or {}
    modalities = arch.get('input_modalities')
    if not isinstance(modalities, list) or not modalities:
        modality = arch.get('modality', '') or ''
        left = modality.split('->')[0]
        modalities = [m.strip() for m in left.split('+') if m.strip()]
    params = set(model.get('supported_parameters', []) or [])
    metadata = {
        "context_window": int(model.get('context_length') or 0),
        "reasoning": bool(params & {"reasoning", "include_reasoning", "reasoning_effort"}),
        "input_modalities": modalities,
    }
    efforts, default_effort = get_reasoning_efforts(model)
    if efforts:
        metadata["reasoning_efforts"] = efforts
        if default_effort:
            metadata["default_effort"] = default_effort
    return metadata

def update_model_list():
    """
    Fetches models from OpenRouter (including the "x-ai" provider entries),
    filters them by capability, and writes the structured models.json.
    """
    logging.info("Starting model list update process...")
    
    try:
        logging.info(f"Fetching model data from {API_URL}...")
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
        raw_models = response.json().get('data', [])
        logging.info(f"Successfully fetched {len(raw_models)} models.")
    except requests.RequestException as e:
        logging.error(f"Failed to fetch model data: {e}")
        return

    # --- Filtering Logic ---
    passed_models = []
    for model in raw_models:
        model_id = model.get('id')
        if not model_id:
            continue

        input_modalities, output_modalities = get_modalities(model)
        if not REQUIRED_INPUT_MODALITIES.issubset(input_modalities):
            continue
        if 'text' not in output_modalities:
            continue
        if output_modalities & FORBIDDEN_OUTPUT_MODALITIES:
            continue
        if DATE_PATTERN.search(model_id) or PREVIEW_PATTERN.search(model_id) or SPECIALIZED_PATTERN.search(model_id):
            continue
        
        provider = model_id.split('/')[0]
        # Skip router endpoints and mirror aliases (~), not real models.
        if provider in BLOCKED_PROVIDERS or provider.startswith('~'):
            continue
        # NOTE: ':free' variants ARE real selectable models (first-class
        # OpenRouter endpoints, just rate-limited). They pass the filters
        # like any other id, but are only routed into OpenRouter groups —
        # never into direct-API lists (native APIs don't know ':free').
        # See the structuring loop below.

        if provider == 'openai' and not re.match(r'openai/gpt-[5-9]', model_id):
            continue

        if provider == 'anthropic':
            match = re.search(r'claude-(\d+(\.\d+)?)', model_id)
            if match:
                try:
                    if float(match.group(1)) <= 4.0:
                        continue
                except (ValueError, IndexError):
                    pass
            
        if model.get('context_length', 0) < MIN_CONTEXT_LENGTH:
            continue

        params = set(model.get('supported_parameters', []) or [])
        if not REQUIRED_PARAMETERS.issubset(params):
            continue
        if not (params & REQUIRED_REASONING_PARAMS):
            continue

        passed_models.append(model)

    logging.info(f"{len(passed_models)} models passed all automated filters.")

    # --- Manual Overrides ---
    passed_model_ids = {m['id'] for m in passed_models}
    final_model_ids = (passed_model_ids | FORCE_INCLUDE) - FORCE_EXCLUDE
    models_by_id = {m['id']: m for m in raw_models}
    final_models = [models_by_id[id] for id in final_model_ids if id in models_by_id]
    logging.info(f"Applied manual overrides. Final model count: {len(final_models)}")

    # --- Structuring Logic ---
    # Explicit entries for native direct providers + friendly group names.
    # Any other provider that passed the capability filters gets an OpenRouter
    # group automatically (display name derived from the provider id).
    PROVIDER_MAP = {
        'google': {'direct_key': 'google', 'display_name': 'Google', 'openrouter_group': 'Google'},
        'openai': {'direct_key': 'openai', 'display_name': 'OpenAI', 'openrouter_group': 'OpenAI'},
        'anthropic': {'direct_key': 'anthropic', 'display_name': 'Anthropic', 'openrouter_group': 'Anthropic'},
        'x-ai': {'direct_key': 'x-ai', 'display_name': 'xAI', 'openrouter_group': 'X-AI'},
        'mistralai': {'openrouter_group': 'Mistral'},
        'meta': {'openrouter_group': 'Meta'},
        'meta-llama': {'openrouter_group': 'Meta'},
        'qwen': {'openrouter_group': 'Qwen'},
        'z-ai': {'openrouter_group': 'Zhipu'},
        'moonshotai': {'openrouter_group': 'Moonshot'},
        'deepseek': {'openrouter_group': 'DeepSeek'},
        'bytedance-seed': {'openrouter_group': 'ByteDance'},
        'xiaomi': {'openrouter_group': 'Xiaomi'},
        'minimax': {'openrouter_group': 'MiniMax'},
        'stepfun': {'openrouter_group': 'StepFun'},
        'inclusionai': {'openrouter_group': 'InclusionAI'},
    }

    def _is_clean_name(name: str) -> bool:
        """Drops :batch variants, dated snapshots and test/build ids from the
        *output* (the input filter works on OpenRouter ids, but direct API
        names derived from canonical_slug can reintroduce dates)."""
        if not name or ":batch" in name:
            return False
        if DATE_PATTERN.search(name):
            return False
        if re.search(r'\b(build|tryme|test)\b', name, re.IGNORECASE):
            return False
        return True

    structured_data = {
        "openrouter": {
            "display_name": "OpenRouter",
            "models_by_provider": {}
        }
    }
    # model_key -> {context_window, reasoning, input_modalities} (built alongside)
    model_metadata = {}

    for model in sorted(final_models, key=lambda m: m.get('id')):
        provider_id = model.get('id').split('/')[0]
        is_free_variant = model.get('id').endswith(':free')
        mapping = PROVIDER_MAP.get(provider_id)
        if not mapping:
            # Auto-group: any capable provider not listed explicitly still
            # gets an OpenRouter group with a derived display name.
            mapping = {'openrouter_group': provider_id.replace('-', ' ').title()}

        # 1. Populate direct provider lists (never ':free' ids —
        # native APIs don't serve that suffix; it is OpenRouter-only).
        direct_key = mapping.get('direct_key')
        if direct_key and not is_free_variant:
            if direct_key not in structured_data:
                structured_data[direct_key] = {
                    "display_name": mapping['display_name'],
                    "models": []
                }

            if provider_id == 'x-ai':
                # xAI direct API names match the OpenRouter id base
                # ("grok-4.5", "grok-4.6", ...) per xAI docs — but NOT the
                # canonical_slug ("grok-4.5-20260708"), which is an OpenRouter
                # date pin invalid on the x.ai API. Hence id-base, not slug.
                direct_api_name = model.get('id').split('/')[-1]
            else:
                direct_api_name = get_direct_api_model_name(model)
            if _is_clean_name(direct_api_name):
                structured_data[direct_key]["models"].append(direct_api_name)
                model_metadata[f"{direct_key}::{direct_api_name}"] = get_model_metadata(model)

        # 2. Populate OpenRouter's nested structure using the standard id
        openrouter_group = mapping.get('openrouter_group')
        if openrouter_group:
            if openrouter_group not in structured_data["openrouter"]["models_by_provider"]:
                structured_data["openrouter"]["models_by_provider"][openrouter_group] = []

            model_name = model.get('id').split('/')[-1]
            if _is_clean_name(model_name):
                structured_data["openrouter"]["models_by_provider"][openrouter_group].append(model_name)
                or_key = f"openrouter::{openrouter_group.lower()}/{model_name.lower().replace(' ', '-')}"
                model_metadata[or_key] = get_model_metadata(model)

    # --- Final Cleanup: Remove duplicates from direct provider lists ---
    for provider_key, provider_data in structured_data.items():
        if 'models' in provider_data:
            provider_data['models'] = sorted(list(set(provider_data['models'])))

    # --- Merge static fallbacks so output is never empty/junk ---
    # Dynamic catalogue ids change constantly; statics guarantee usability.
    for direct_key, fallback_models in STATIC_DIRECT_MODELS.items():
        if direct_key not in structured_data:
            structured_data[direct_key] = {"display_name": direct_key.title(), "models": []}
        merged = sorted(set(structured_data[direct_key].get("models", [])) | set(fallback_models))
        structured_data[direct_key]["models"] = merged
    or_groups = structured_data["openrouter"]["models_by_provider"]
    for grp, fallback_models in STATIC_OPENROUTER_GROUPS.items():
        or_groups[grp] = sorted(set(or_groups.get(grp, [])) | set(fallback_models))

    # Hugging Face Router section is fully static (separate catalogue).
    structured_data["huggingface"] = {
        "display_name": "Hugging Face",
        "models": sorted(STATIC_HUGGINGFACE_MODELS),
    }

    # --- Validate before writing: refuse junk (fewer than 2 usable providers) ---
    usable = 0
    for _pid, info in structured_data.items():
        total = len(info.get("models", []) or [])
        total += sum(len(v) for v in (info.get("models_by_provider", {}) or {}).values() if isinstance(v, list))
        if total > 0:
            usable += 1
    if usable < 2:
        logging.error(f"Generated data looks like junk ({usable} usable providers), NOT writing {OUTPUT_FILE}.")
        return

    # --- Per-mode capabilities block ---
    # Built after validation (it is not a provider entry). Keys are exactly the
    # ids Logos stores, so the app can do an O(1) membership test per group.
    mode_caps = {'general': [], 'vision': []}
    for provider_id, info in structured_data.items():
        if provider_id == 'openrouter':
            # OpenRouter is a router: the app does NOT drive computer use
            # through it, so its models are general-only even when the
            # underlying model (Claude/GPT/Gemini) would be vision-capable.
            for group, models in (info.get('models_by_provider') or {}).items():
                for name in models:
                    mode_caps['general'].append(f"openrouter::{group.lower()}/{name.lower().replace(' ', '-')}")
        else:
            for name in (info.get('models') or []):
                key = f"{provider_id}::{name}"
                mode_caps['general'].append(key)
                if _is_vision_capable(provider_id, name):
                    mode_caps['vision'].append(key)
    structured_data['mode_capabilities'] = {m: sorted(set(v)) for m, v in mode_caps.items()}
    logging.info("mode_capabilities: " + ", ".join(f"{m}={len(v)}" for m, v in mode_caps.items()))

    # --- Per-model metadata block ---
    # context window + reasoning + input modalities, keyed exactly like
    # mode_capabilities so Logos can look any model up in O(1). Models that
    # only come from the static fallbacks (not in the catalogue) are absent;
    # Logos applies a safe default window for those.
    structured_data['model_metadata'] = model_metadata
    logging.info(f"model_metadata: {len(model_metadata)} entries")

    # --- Save to File ---
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2)
        logging.info(f"Successfully saved structured model list to {OUTPUT_FILE}")
    except IOError as e:
        logging.error(f"Failed to write to output file {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    update_model_list()
