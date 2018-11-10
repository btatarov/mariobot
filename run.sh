#!/bin/bash
if [ "$(uname)" == "Darwin" ]; then
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    caffeinate -i python mariobot.py
else
    python mariobot.py
fi
