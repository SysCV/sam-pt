"""
This script cleans-up the original TapNet checkpoint by removing objects
that require `import tapnet` to work. The cleaned checkpoint saves only
the weights and removes the optimizer state. The cleaned checkpoint can
be used within SAM-PT.

Note that we provide a link to the cleaned checkpoint in the
documentation and that you might not need to run this script yourself.

Usage:
1. Clone the [TapNet repository](https://github.com/deepmind/tapnet) and
   checkout the commit `ba1a8c8f2576d81f7b8d69dbee1e58e8b7d321e1`.
2. Setup the TapNet environment.
3. Run this script one level above the TapNet repository (i.e., not
   within the TapNet repository, but within its parent directory). For
   that, navigate to the parent directory of TapNet repository (`cd ..`)
   and set the PYTHONPATH environment variable
   (```export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH```).

Run the script from the command line with the following arguments:
- --input: The path to the original TapNet checkpoint file. 
- --output: The path where the cleaned checkpoint file will be saved.

For example:
```bash
python script_name.py \
  --input "./models/tapnet_ckpts/open_source_ckpt/checkpoint.npy" \
  --output "./models/tapnet_ckpts/open_source_ckpt/checkpoint_wo_optstate.npy"
```
"""

import argparse
import numpy as np
import tensorflow as tf


def clean_checkpoint(input_path, output_path):
    # Load the original checkpoint file.
    checkpoint = np.load(input_path, allow_pickle=True).item()

    print(checkpoint.keys())
    # dict_keys(['params', 'state', 'opt_state', 'global_step'])

    # Create a new dictionary without the 'opt_state' and 'global_step'.
    checkpoint_wo_optstate = {
        "params": checkpoint["params"],
        "state": checkpoint["state"],
    }

    # Save the cleaned checkpoint file.
    with tf.io.gfile.GFile(output_path, 'wb') as fp:
        np.save(fp, checkpoint_wo_optstate)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="The path to the original TapNet checkpoint file.")
    parser.add_argument("--output", help="The path where the cleaned checkpoint file will be saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    clean_checkpoint(args.input, args.output)
