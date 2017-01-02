EPOCH=100
python3 cnn25d.py -i test.small.h5 -e $EPOCH --no-binary -o 2.5D_model --train --test --no-plot
python3 cnn25d.py -i test.small.hex.h5 -e $EPOCH --no-binary -o hex_model --hex --train --test --no-plot
python3 plot_history.py -i hex_model.h5 2.5D_model.h5