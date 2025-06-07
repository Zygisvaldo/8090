#!/bin/bash

# Check if exactly three arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>" >&2
    exit 1
fi

# Call calculate_reimbursement.py and extract the numeric prediction
python3 calculate_reimbursement.py "$1" "$2" "$3" | awk '/Predicted reimbursement/ {print $3}'
