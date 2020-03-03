echo ""
echo "#####################################"
echo "#       TEST prediction output      #"
echo "#####################################"

feat_dim_mfcc=200
feat_dim_asr=8053

mkdir -p test_pred
for event in P003; do
    python2 scripts/test_svm.py mfcc_pred/svm.$event.3.model "kmeans/" $feat_dim_mfcc test_pred/${event}_test_mfcc.lst || exit 1;
    python2 scripts/test_svm.py asr_pred/svm.$event.3.model "asrfeat/" $feat_dim_asr test_pred/${event}_test_asr.lst || exit 1;
done
