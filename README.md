# MarioBot AI

Simple self-learning AI based on NEAT that plays Super Mario Bros. for the NES.

## Requirements
* Python 3.6
* FFmpeg (for rendering movies from playback records)

## Installation
    git clone https://github.com/btatarov/mariobot.git
    cd marioibot
    python3 -m venv .
    source bin/activate
    pip install -r requirements.txt

    # put your mario.nes ROM in ./game/ named rom.nes
    mkdir lib/python3.6/site-packages/retro/data/stable/Mario-Nes/
    cp game/* lib/python3.6/site-packages/retro/data/stable/Mario-Nes/

## Running
Use `./run.sh` or `python mariobot.py`

## Render playback movies
Use `./convert-movies.sh` to convert all records or:
`python -m retro.scripts.playback_movie records/directory_uuid/your_record.bk2`
