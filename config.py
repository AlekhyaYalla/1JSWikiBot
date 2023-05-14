#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "ad9259c3-7e3c-4bfd-bf24-041d8b979c90")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "AlekhyaAjayYalla@123")
    APP_TYPE = os.environ.get("MicrosoftAppType", "MultiTenant")
    APP_TENANT_ID = os.environ.get("MicrosoftAppTenantId", "")
