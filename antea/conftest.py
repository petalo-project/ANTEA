import os
import pytest

from collections import namedtuple

db_data  = namedtuple('db_data', 'detector nsipms conf_label')


@pytest.fixture(scope = 'session')
def ANTEADIR():
    return os.environ['ANTEADIR']


@pytest.fixture(scope = 'session')
def ANTEADATADIR(ANTEADIR):
    return os.path.join(ANTEADIR, "testdata/")


@pytest.fixture(scope = 'session')
def config_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('configure_tests')


@pytest.fixture(scope = 'session')
def output_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('output_files')


@pytest.fixture(scope='session',
                params=[db_data('petalo' ,    3500, 'P7R195Z140mm'),
                        db_data('petalo' ,     128, 'PB')],
                ids=["petit", "petbox"])
def db(request):
    return request.param


@pytest.fixture(scope='session',
                params=[db_data('petalo' ,  104528, 'P7R420Z1950mm'),
                        db_data('petalo' ,   99802, 'P7R400Z1950mm'),
                        db_data('petalo' ,  102304, 'P7R410Z1950mm')],
                ids=["fullring4cm", "fullring2cm", "sim-only"])
def db_sim_only(request):
    return request.param
