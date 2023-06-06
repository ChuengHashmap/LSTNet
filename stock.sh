#!/bin/bash

gpu_list=('1p' '2p' '3p' '4p' '5p' '6p' '7p')

get_gpu(){
  while true;
  do
    for ((i=0; i<=6; i++));
    do
      a=$(nvidia-smi | \
      grep -E "[0-9]+MiB\s*/\s*[0-9]+MiB" | sed -n ${gpu_list[i]} | \
      awk '{print ($9" "$11)}' | \
      sed "s/\([0-9]\{1,\}\)MiB \([0-9]\{1,\}\)MiB/\1 \2/" | \
      awk '{print ($2 - $1)/$2}')
      min=0.5;
      if [ `expr $a \> $min` -eq 0 ];
      then
        continue
      else
        echo $i && return 2
      fi
    done
    sleep 10s
  done
}

b_ses=(16 32 64)
hid_CNNs=(30 50 70 90 100)
hid_RNNs=(20 50 70 90 100)
dropouts=(0.3 0.4 0.5 0.6)
GRU_layerses=(1 2 3 4)
lrs=(0.001 0.005 0.01)
for lr in ${lrs[*]};
do
  for hid_CNN in ${hid_CNNs[*]};
  do
    for hid_RNN in ${hid_RNNs[*]};
    do
      for dropout in ${dropouts[*]};
      do
        for b_s in ${b_ses[*]};
          do
            for gru_layers in ${GRU_layerses[*]};
            do
              gpu=$(echo $(get_gpu))
              python main.py --gpu $gpu --data data/exchange_rate.txt --save save/exchange_rate.pt \
              --hidCNN $hid_CNN --hidRNN $hid_RNN \
              --L1Loss False --output_fun None \
              --lr $lr --batch_size $b_s --dropout $dropout --GRU_layers $gru_layers &
              sleep 10s
            done
          done
      done
    done
  done
done
