from django.urls import path
from django.conf import settings ##new add
from django.conf.urls. static import static ## new add
# from . import views
from .views import (
    analysis_views,
    download_views,
    help_views,
    home_views,
    predict_result,
    predict_views,
    preview_views,
    validated_analysis_views,
)


urlpatterns=[
    path('hello_world',home_views.hello_world),
    path('home',home_views.home),

    # path('analysis_oneclick',analysis_views.analysis_oneclick),
    # path('analysis_perpara',analysis_views.analysis_perpara),
    path('analysis', analysis_views.get_analysis_page),
    path('analysis/methods.json', analysis_views.method_registry),
    path('predict', predict_views.get_predict_page),

    path('predict_preview',predict_views.predict_preview),
    path('predict_result/<str:projectid>', predict_result.predict_results),

    path('help', help_views.get_help_page),
    path('download/<str:fname>', download_views.download_sample_data),
    path('preview', preview_views.preview_result),
    # model download
    path('model_download/<str:projectid_model>', download_views.download_model),
    # classification
    # one click result
    # path('classification_oc_result/<str:projectid>', classification_oc_result_views.result),
    path('classification_oc_result/<str:projectid>', validated_analysis_views.result),
    path('classification_oc_result_ws/<str:projectid>', validated_analysis_views.legacy_disabled),

    # custom parameter result
    # path('classification_cp_result/<str:projectid>', classification_cp_result_views),
    # path('classification_cp_result_ws/<str:projectid>', classification_cp_result_views.result),
    path('classification_cp_result_ws/<str:projectid>', validated_analysis_views.legacy_disabled),
    path('classification_cp_result/<str:projectid>', validated_analysis_views.result),
    path('classification_cp_result/prev/<str:projectid_paramd5>', validated_analysis_views.previous_result),

    # regression
    # path('regression_oc_result/<str:projectid>', regression_oc_result_views.regression_oc_result),
    # path('regression_cp_result/<str:projectid>', regression_cp_result_views.regression_cp_result),
    path('regression_oc_result/<str:projectid>', validated_analysis_views.result),
    path('regression_oc_result_ws/<str:projectid>', validated_analysis_views.legacy_disabled),

    path('regression_cp_result/<str:projectid>', validated_analysis_views.result),
    path('regression_cp_result_ws/<str:projectid>', validated_analysis_views.legacy_disabled),
    path('regression_cp_result/prev/<str:projectid_paramd5>', validated_analysis_views.previous_result),
    # survival
    # path('survival_oc_result/<str:projectid>', survival_oc_result_views.survival_oc_result),
    # path('survival_cp_result/<str:projectid>', survival_cp_result_views.survival_cp_result),
    path('survival_oc_result/<str:projectid>', validated_analysis_views.result),
    path('survival_oc_result_ws/<str:projectid>', validated_analysis_views.legacy_disabled),
    path('survival_cp_result/<str:projectid>', validated_analysis_views.result),
    path('survival_cp_result_ws/<str:projectid>', validated_analysis_views.legacy_disabled),

    path('survival_cp_result/prev/<str:projectid_paramd5>', validated_analysis_views.previous_result),

    path('task/<str:projectid>/result.json', validated_analysis_views.result_json),
    path('task/<str:projectid>/model/<str:filename>', validated_analysis_views.model_bundle),
    path('task/<str:projectid>/artifact/<str:filename>', validated_analysis_views.prediction_artifact),
    path('task/<str:projectid>/delete', validated_analysis_views.delete_project),

    #websocket test
    # path('vue_analysis', websocket.get_vue_analysis_page), ####vue
    # path('acceptsocket', websocket.test_websocket2),
    # path('vue_analysis', websocket.test_websocket2,name='test_websocket'),
    # path('test_websocket', websocket.test_websocket2, name='test_websocket'),
    # path('test_websocket_client', websocket.test_websocket_client , name='test_websocket_client'),

] + static (settings.STATIC_URL, document_root = settings.STATIC_ROOT)
