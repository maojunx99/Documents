#!/usr/bin/env bash
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/_stage.sh" regenerate "$@"
