Modify these to your needs to debug live mode.

Use `setbreak.txt` to set any breakpoints, and other things before gdb
runs. Modify `test-estimate-ngram` to set any needed arugments passed to
estimate-ngram.

Then, if debugging through UnnaturalCode, do this:

    export ESTIMATENGRAM=$PWD/test-estimate-ngram

Replace `$PWD` with the full path to this directory as necessary.
