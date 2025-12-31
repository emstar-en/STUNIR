#!/usr/bin/env bash
# STUNIR Shell-Native JSON Writer
# A simple, dependency-free JSON builder using string concatenation.

JSON_BUFFER=""

json_init() {
    JSON_BUFFER=""
}

json_start_object() {
    JSON_BUFFER="${JSON_BUFFER}{"
}

json_end_object() {
    JSON_BUFFER="${JSON_BUFFER}}"
}

json_start_array() {
    JSON_BUFFER="${JSON_BUFFER}["
}

json_end_array() {
    JSON_BUFFER="${JSON_BUFFER}]"
}

json_key_start_array() {
    local key=$1
    JSON_BUFFER="${JSON_BUFFER}\"$key\":["
}

json_key_val() {
    local key=$1
    local val=$2
    # Basic escaping for quotes
    val=${val//\"/\\\"}
    JSON_BUFFER="${JSON_BUFFER}\"$key\":\"$val\""
}

json_add_comma() {
    JSON_BUFFER="${JSON_BUFFER},"
}
