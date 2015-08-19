make checkwaf
python waf configure --prefix=$PREFIX
python waf build
python waf install
pushd python/
python setup.py build
python setup.py install
popd
