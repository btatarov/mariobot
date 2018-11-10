#!/bin/bash
find ./records/ -name "*.bk2" -exec python -m retro.scripts.playback_movie {} \;
