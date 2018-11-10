#!/bin/bash
find ./records/ -name "*.bk2" -exec python lib/python3.6/site-packages/retro/scripts/playback_movie.py {} \;
