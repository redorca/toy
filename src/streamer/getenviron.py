#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

if os.getenv("TB_ACCOUNT"):
    print(
        " ips options setting ACCOUNT from bash environment variable TB_ACCOUNT",
        os.getenv("TB_ACCOUNT"),
    )
    account = acct = ACCOUNT = os.getenv("TB_ACCOUNT")
if os.getenv("TB_HOST"):
    print(
        " ips options setting HOST from bash environment variable TB_HOST",
        os.getenv("TB_HOST"),
    )
    host = HOST = os.getenv("TB_HOST")
if os.getenv("TB_PORT"):
    print(
        " ips options setting PORT from bash environment variable TB_PORT",
        os.getenv("TB_PORT"),
    )
    port = PORT = os.getenv("TB_PORT")

if "host" not in vars() and "host" not in globals():
    from bot import conf

    host = conf.host
    port = conf.port


comment = "this bit of code copied in from an email, not yet connected to this project"
del comment
