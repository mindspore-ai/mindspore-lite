# Security for MindSpore Lite

## Security Risk Description

When run a model using MindSpore Lite, the value from the model will be read and used as the parameter or input of a operator, if the value read from the model is invalid, it may cause unexpected result. For example, if the invalid value is used as the offset of a vector, it may cause your app run into segmentation fault issue.

## Security Usage Suggestions

1. Make sure your model is well verified and protected.
2. The exception catching mechanism of C++ is an effective method to improve robustness of your app, consider adding code to catch exception when calling the MindSpore Lite API, as exception will be raised in some case such as the example mentioned in the risk description above.
