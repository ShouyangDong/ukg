UKG_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && $pwd)"
TVM_PATH="${UKG_HOME}/3rdparty/tvm"

export PYTHONPATH=$UKG_HOME/python:$TVM_PATH/python:${PYTHONPATH}
export TVM_LOG_DEBUG=1
