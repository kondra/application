#!/bin/bash

tr '[:upper:]' '[:lower:]' < $1 | python translit.py | tr '[:punct:]' ' ' | tr '[:space:]' '\n' | tr -cd 'a-z\n' | tr -s '\n' > $2
#tr '[:upper:]' '[:lower:]' < $1 | tr '[:punct:]' ' ' | tr '[:space:]' '\n' | python translit.py | tr -cd 'a-z ' > $2
