#!/bin/bash
find ./record/ -name "*.bk2" -exec python movie.py {} \;
