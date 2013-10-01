#!/bin/bash
valgrind --leak-check=full "`dirname $0`/.libs/estimate-ngram" "$@"
