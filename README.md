# Main Quest 11 AutoInt+ Model 실험 요약
Main Quest 11 에서 AutoInt + 모델의 성능 향상을 위해  3가지 실험을 함.   test 1에서  hidden layer를 50으로 늘려서 성능이 향상되어 test 2에서 70까지 올려봤으나 성능이 떨어짐.  hidden layer를 너무 높일 필요는 없음.     test 3에서 hidden layer를 50으로 놓고 epoch을 3으로 줄여봤는데  ndgc (Normalized Discounted Cumulative Gain)은 올라갔으나, hit rate이 떨어짐.  충분한 학습은 필요한듯하여  epoch을 5로 돌리려 함.

# Main Quest 11 Streamlit 작성시 문제점
마지막 단계인 영화추천 단계에서 코드가 작동하지 않았음.  data가 비어 있지는 않는데도, 모델이 추천할 영화를 찾지 못하고 있음.   모델이 예측한 상위 20개 영화의 평점을 확인해보려 했으나, 코드가 작동하지 않았음.   일단 여기서 프로젝트 마무리하고 차후 다시 문제를 풀어볼 예정임.

# Streamlit 작동영상
[영상 보러 가기 - YouTube](https://www.youtube.com//video/bgtLMP20I2g/edit)
