"""Public upload admission tests: reject before any multipart data is stored."""
from types import SimpleNamespace
import re
from unittest.mock import Mock, patch

from django.http import HttpResponse
from django.test import Client, RequestFactory, SimpleTestCase, override_settings

from .middleware import MalerUploadCapacityMiddleware


class ProbabilityCalibrationReproducibilityTests(SimpleTestCase):
    def test_registry_svm_probabilities_ignore_unrelated_numpy_rng_state(self):
        import numpy as np
        from .model_registry import build_estimator
        rng = np.random.RandomState(813)
        X = rng.normal(size=(80, 8))
        y = np.where(X[:, 0] + rng.normal(size=80) > 0, 'case', 'control')
        original = np.random.get_state()
        try:
            np.random.seed(1)
            first = build_estimator('classification', 'svm')[1].fit(X, y)
            np.random.seed(997)
            second = build_estimator('classification', 'svm')[1].fit(X, y)
            np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))
        finally:
            np.random.set_state(original)


@override_settings(MALER_PUBLIC_UPLOADS_ENABLED=True, MALER_MIN_FREE_TASK_BYTES=128,
                   MALER_TASK_ROOT='/private-tasks')
class PublicStorageAdmissionTests(SimpleTestCase):
    def setUp(self):
        self.downstream = Mock(return_value=HttpResponse('ok'))
        self.middleware = MalerUploadCapacityMiddleware(self.downstream)

    def test_low_capacity_rejects_both_upload_routes_before_view(self):
        for route in ['/maler/preview','/maler/predict_preview']:
            request = RequestFactory().post(route, data=b'x'*50, content_type='application/octet-stream')
            with patch('mlserver.middleware.shutil.disk_usage', return_value=SimpleNamespace(free=160)):
                self.assertEqual(self.middleware(request).status_code,507)
        self.downstream.assert_not_called()

    def test_available_capacity_and_information_pages_continue(self):
        request = RequestFactory().post('/maler/preview', data=b'x'*10, content_type='application/octet-stream')
        with patch('mlserver.middleware.shutil.disk_usage', return_value=SimpleNamespace(free=512)):
            self.assertEqual(self.middleware(request).status_code,200)
        with patch('mlserver.middleware.shutil.disk_usage', side_effect=AssertionError('Information pages must remain available')):
            self.assertEqual(self.middleware(RequestFactory().get('/maler/home')).status_code,200)
        self.assertEqual(self.downstream.call_count,2)

    def test_unavailable_storage_and_invalid_length_fail_without_view(self):
        request = RequestFactory().post('/maler/preview', data=b'x', content_type='application/octet-stream')
        with patch('mlserver.middleware.shutil.disk_usage', side_effect=OSError('unavailable')):
            self.assertEqual(self.middleware(request).status_code,503)
        request.META['CONTENT_LENGTH']='invalid'
        self.assertEqual(self.middleware(request).status_code,400)
        self.downstream.assert_not_called()


@override_settings(MALER_PUBLIC_UPLOADS_ENABLED=True, MALER_MIN_FREE_TASK_BYTES=0)
class PublicHttpsCsrfTests(SimpleTestCase):
    def test_referrer_policy_hides_task_paths_but_allows_origin(self):
        client=Client(enforce_csrf_checks=True)
        for page in ['home','analysis','predict','help']:
            response=client.get('/maler/'+page,secure=True,HTTP_HOST='www.inbirg.com')
            self.assertEqual(response['Referrer-Policy'],'strict-origin')

    def test_https_forms_require_csrf_and_matching_origin(self):
        client=Client(enforce_csrf_checks=True)
        response=client.get('/maler/predict',secure=True,HTTP_HOST='www.inbirg.com')
        token=re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"',response.content.decode()).group(1)
        form={'csrfmiddlewaretoken':token,'select_model':'invalid-task'}
        # A 400 comes from form validation, proving HTTPS CSRF passed.
        response=client.post('/maler/predict_preview',form,secure=True,
                             HTTP_HOST='www.inbirg.com',HTTP_REFERER='https://www.inbirg.com/')
        self.assertEqual(response.status_code,400)
        for referer in [None,'https://unrelated.example/']:
            headers={'HTTP_HOST':'www.inbirg.com'}
            if referer is not None:headers['HTTP_REFERER']=referer
            self.assertEqual(client.post('/maler/predict_preview',form,secure=True,**headers).status_code,403)
        self.assertEqual(client.post('/maler/predict_preview',{'select_model':'invalid-task'},
                        secure=True,HTTP_HOST='www.inbirg.com',HTTP_REFERER='https://www.inbirg.com/').status_code,403)

