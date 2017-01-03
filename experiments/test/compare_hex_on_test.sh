cd ../..
EPOCH=200
MODELPATH=models/test

# BASIC
python3 cnn25d.py -i test/test.small.r5.h5 -e $EPOCH --no-binary -o $MODELPATH/2.5D --train --test --no-plot
python3 cnn25d.py -i test/test.small.r5.hex.h5 -e $EPOCH --no-binary -o $MODELPATH/HEX --hex --train --test --no-plot

# RESIDUAL
python3 cnn25d.py -i test/test.small.r5.h5 -e $EPOCH --no-binary -o $MODELPATH/2.5D.res --train --test --no-plot --cnntype residual
python3 cnn25d.py -i test/test.small.r5.hex.h5 -e $EPOCH --no-binary -o $MODELPATH/HEX.res --hex --train --test --no-plot --cnntype residual

python3 plot_history.py -i $MODELPATH/HEX.h5 $MODELPATH/2.5D.h5 $MODELPATH/2.5D.res.h5 $MODELPATH/HEX.res.h5 -o $MODELPATH/compare_all.eps