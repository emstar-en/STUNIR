#!/bin/sh
# STUNIR Shell-Native JSON Builder
# Uses printf for reliable formatting across shells.

_JSON_FILE=""
_JSON_FIRST_ITEM=1

json_init() {
    _JSON_FILE="$1"
    # We don't write { here because manifest.sh writes the header manually.
    # This function just sets the file path variable.
    # If we wanted to start a fresh object we would, but manifest.sh usage is specific.
    # Actually, manifest.sh calls json_init just to set the variable.
}

_json_comma() {
    if [ "$_JSON_FIRST_ITEM" -eq 0 ]; then
        printf "," >> "$_JSON_FILE"
    fi
    _JSON_FIRST_ITEM=0
}

json_obj_start() {
    _json_comma
    printf "\n  {" >> "$_JSON_FILE"
    _JSON_FIRST_ITEM=1
}

json_obj_end() {
    printf "\n  }" >> "$_JSON_FILE"
    _JSON_FIRST_ITEM=0
}

json_key_str() {
    # Usage: json_key_str "key" "value"
    key="$1"
    val="$2"

    _json_comma

    # Escape backslashes and double quotes
    # We use a safe sed pattern. 
    # Note: We must escape backslashes first, then quotes.
    safe_val=$(printf '%s' "$val" | sed 's/\\/\\\\/g; s/"/\\"/g')

    # Print "key": "value"
    # We add a newline before the key for readability
    printf "\n    \"%s\": \"%s\"" "$key" "$safe_val" >> "$_JSON_FILE"
}

json_key_int() {
    # Usage: json_key_int "key" 123
    key="$1"
    val="$2"

    _json_comma
    printf "\n    \"%s\": %s" "$key" "$val" >> "$_JSON_FILE"
}
