passwords = {
    "login": "password",
    "account": "password",
    # e.g.
    "PU123456-890": "shhh_don't_tell_anyone",
}

# copy this template to secrets.py
# fill in the secrets
# in other files that need the secrets:
#
# from streamer import secrets
# passwords = secrets.passwords
# my_password = passwords['login']
