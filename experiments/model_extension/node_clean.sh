#!/bin/bash
# per-node cleanup (run as a file so no command line carries the patterns it kills)
pkill -f 'train_llama_e2e_amp[67]'; pkill -f 'training/g_trai[n].py'; pkill -f 'torchru[n]'; sleep 2
find /tmp/neuron_cache -name '*.lock' -delete 2>/dev/null
for d in $(find /tmp/neuron_cache -maxdepth 2 -type d -name 'MODULE_*' 2>/dev/null); do [ -f $d/model.neff ] || rm -rf $d; done
echo "$(hostname): procs=$(pgrep -af 'torchru[n]|train_llam[a]|g_trai[n]' | wc -l) fuser=$(sudo fuser /dev/neuron* 2>/dev/null | wc -w) locks=$(find /tmp/neuron_cache -name '*.lock' 2>/dev/null | wc -l) modules=$(find /tmp/neuron_cache -maxdepth 2 -type d -name 'MODULE_*' 2>/dev/null | wc -l)"
