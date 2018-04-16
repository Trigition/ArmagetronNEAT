#!/bin/bash

ffmpeg -f image2 -pattern_type glob -framerate 60 -i 'images/*.jpg' sim.mpeg
