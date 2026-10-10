import json
import re
import requests
import logging
from pathlib import Path
import os

# --- Configuration: All filtering rules are defined here for easy tuning ---

# The URL for fetching the list of all models from the OpenRouter API.
API_URL = "https://openrouter.ai/api/v1/models"
XAI_API_URL = "https://api.x.ai/v1/models"

# Define project root and paths. In this new repo, the script is at the root.
PROJECT_ROOT = Path(__file__).parent
OUTPUT_FILE = PROJECT_ROOT / "models.json"

# This map now serves as a reliable fallback if the x.ai API cannot be reached
# or if no API key is provided.
XAI_DIRECT_API_FALLBACK_MAP = {
    "grok-4": ["grok-4", "grok-4-0709"],
    "grok-4-fast": ["grok-4-fast-reasoning", "grok-4-fast-non-reasoning"],
    "grok-4-1-fast-reasoning": ["grok-4-1-fast-reasoning"],
    "grok-4-fast-reasoning": ["grok-4-fast-reasoning"],
}

# --- Step 1: Technical Filters ---
# We want multimodal models that can process text and images. OpenRouter used
# to report exactly "text+image->text", but flagship models now report extended
# modalities like "text+image+file->text" or "text+image+file+audio+video->text".
# So we check the modality *prefix* instead of exact equality.
REQUIRED_MODALITY_PREFIX = "text+image"

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
# A minimum context length to filter out older or less capable models.
MIN_CONTEXT_LENGTH = 32000
# The model MUST support these parameters to be controllable by our
# application for structured output and tool use. NOTE: "reasoning" was
# dropped — it killed almost every flagship model (gemini/gpt-5/gemma have
# tools+tool_choice but not always "reasoning"), and reasoning is a
# nice-to-have, not a requirement for tool calling.
REQUIRED_PARAMETERS = {
    "tool_choice",
    "tools",
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
FORCE_EXCLUDE = set()

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

def get_xai_api_key():
    """
    Reads the x.ai API key from the environment variable.
    """
    api_key = os.getenv("XAI_API_KEY")
    if api_key:
        logging.info("Found x.ai API key in environment variable.")
    else:
        logging.warning("XAI_API_KEY environment variable not found.")
    return api_key

def fetch_xai_models_from_api(api_key):
    """
    Fetches the list of models directly from the x.ai API.
    Returns a list of model IDs on success, or None on failure.
    """
    if not api_key:
        return None
    logging.info("Attempting to fetch model list from x.ai API...")
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(XAI_API_URL, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json().get('data', [])
        model_ids = [model['id'] for model in data]
        logging.info(f"Successfully fetched {len(model_ids)} models from x.ai API.")
        return model_ids
    except requests.RequestException as e:
        logging.warning(f"Could not fetch models from x.ai API: {e}. No direct x.ai models will be added.")
        return None

def update_model_list():
    """
    Fetches models from OpenRouter, filters them, and resolves x.ai model names.
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

        if not model.get('architecture', {}).get('modality', '').startswith(REQUIRED_MODALITY_PREFIX):
            continue
        if DATE_PATTERN.search(model_id) or PREVIEW_PATTERN.search(model_id) or SPECIALIZED_PATTERN.search(model_id):
            continue
        
        provider = model_id.split('/')[0]
        # Skip router endpoints, mirror aliases (~) and :free variants —
        # not real selectable models.
        if provider in BLOCKED_PROVIDERS or provider.startswith('~'):
            continue
        if model_id.endswith(':free'):
            continue

        if provider == 'openai' and not model_id.startswith('openai/gpt-5'):
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

        if not REQUIRED_PARAMETERS.issubset(set(model.get('supported_parameters', []))):
            continue

        passed_models.append(model)

    logging.info(f"{len(passed_models)} models passed all automated filters.")

    # --- Manual Overrides ---
    passed_model_ids = {m['id'] for m in passed_models}
    final_model_ids = (passed_model_ids | FORCE_INCLUDE) - FORCE_EXCLUDE
    models_by_id = {m['id']: m for m in raw_models}
    final_models = [models_by_id[id] for id in final_model_ids if id in models_by_id]
    logging.info(f"Applied manual overrides. Final model count: {len(final_models)}")

    # --- Dynamic x.ai Model Resolution ---
    xai_api_key = get_xai_api_key()
    live_xai_models = fetch_xai_models_from_api(xai_api_key)
    
    # Get the base names of x.ai models that passed our filters
    approved_xai_base_names = {
        m['id'].split('/')[-1] for m in final_models if m['id'].startswith('x-ai/')
    }
    
    xai_direct_models = []
    if live_xai_models:
        # Match live models against approved base names
        for live_model in live_xai_models:
            for base_name in approved_xai_base_names:
                if live_model.startswith(base_name):
                    xai_direct_models.append(live_model)
                    break
    else:
        logging.info("No x.ai API key found or API fetch failed. Using fallback map for direct x.ai models.")
        for base_name in approved_xai_base_names:
            xai_direct_models.extend(XAI_DIRECT_API_FALLBACK_MAP.get(base_name, []))

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

    for model in sorted(final_models, key=lambda m: m.get('id')):
        provider_id = model.get('id').split('/')[0]
        mapping = PROVIDER_MAP.get(provider_id)
        if not mapping:
            # Auto-group: any capable provider not listed explicitly still
            # gets an OpenRouter group with a derived display name.
            mapping = {'openrouter_group': provider_id.replace('-', ' ').title()}

        # 1. Populate direct provider lists
        direct_key = mapping.get('direct_key')
        if direct_key:
            if direct_key not in structured_data:
                structured_data[direct_key] = {
                    "display_name": mapping['display_name'],
                    "models": []
                }
            
            if provider_id == 'x-ai':
                # x.ai models were resolved pre-loop
                structured_data[direct_key]["models"] = sorted(list(set(xai_direct_models)))
            else:
                direct_api_name = get_direct_api_model_name(model)
                if _is_clean_name(direct_api_name):
                    structured_data[direct_key]["models"].append(direct_api_name)

        # 2. Populate OpenRouter's nested structure using the standard id
        openrouter_group = mapping.get('openrouter_group')
        if openrouter_group:
            if openrouter_group not in structured_data["openrouter"]["models_by_provider"]:
                structured_data["openrouter"]["models_by_provider"][openrouter_group] = []

            model_name = model.get('id').split('/')[-1]
            if _is_clean_name(model_name):
                structured_data["openrouter"]["models_by_provider"][openrouter_group].append(model_name)

    # --- Final Cleanup: Remove duplicates from direct provider lists ---
    for provider_key, provider_data in structured_data.items():
        if provider_key != 'x-ai' and 'models' in provider_data:
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

    # --- Save to File ---
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2)
        logging.info(f"Successfully saved structured model list to {OUTPUT_FILE}")
    except IOError as e:
        logging.error(f"Failed to write to output file {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    update_model_list()
