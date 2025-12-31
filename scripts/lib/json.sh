#!/bin/sh
# STUNIR Shell-Native JSON Builder
# A simple, dependency-free append-only JSON writer.

_JSON_FILE=""
_JSON_FIRST_ITEM=1
_JSON_DEPTH=0

json_init() {
    _JSON_FILE="$1"
    echo "" > "$_JSON_FILE"
    _JSON_FIRST_ITEM=1
    _JSON_DEPTH=0
}

_json_comma() {
    if [ "$_JSON_FIRST_ITEM" -eq 0 ]; then
        echo "," >> "$_JSON_FILE"
    fi
    _JSON_FIRST_ITEM=0
}

json_obj_start() {
    _json_comma
    echo "{" >> "$_JSON_FILE"
    _JSON_FIRST_ITEM=1
    _JSON_DEPTH=$((_JSON_DEPTH + 1))
}

json_obj_end() {
    echo "" >> "$_JSON_FILE"
    echo "}" >> "$_JSON_FILE"
    _JSON_FIRST_ITEM=0
    _JSON_DEPTH=$((_JSON_DEPTH - 1))
}

json_arr_start() {
    _json_comma
    echo "[" >> "$_JSON_FILE"
    _JSON_FIRST_ITEM=1
}

json_arr_end() {
    echo "" >> "$_JSON_FILE"
    echo "]" >> "$_JSON_FILE"
    _JSON_FIRST_ITEM=0
}

json_key_str() {
    # Usage: json_key_str "key" "value"
    _json_comma
    # Basic escaping for quotes and backslashes
    val=$(echo "$2" | sed 's/\\/\\\\/g; s/"/\\"/g')
    echo "\"\"$1\": \"$val\"" >> "$_JSON_FILE"
}

json_key_int() {
    # Usage: json_key_int "key" 123
    _json_comma
    echo "\"\"$1\": $2" >> "$_JSON_FILE"
}
