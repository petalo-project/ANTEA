#!/usrbin/env bash

COMMAND=$1
ARGUMENT=$2

## Interpret meaning of command line argument depending on which
## function will receive it.

function run_tests {
    # Run the test suite
    pytest --instafail --no-success-flaky-report
}


function download_test_db {
    echo Downloading database
    python $ANTEADIR/database/download.py $ARGUMENT
}


THIS=manage.sh

## Main command dispatcher

case $COMMAND in
    run_tests)                       run_tests ;;
    run_tests_par)                   run_tests_par ;;
    download_test_db)                download_test_db ;;

    *) echo Unrecognized command: ${COMMAND}
       echo
       echo Usage:
       echo
       echo "bash   $THIS run_tests"
       echo "bash   $THIS download_test_db"
       ;;
esac
