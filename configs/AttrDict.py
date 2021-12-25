class AttrDict(dict):
    def __init__(self, *args, **kwargs):

        super(AttrDict, self).__init__(kwargs)

    def __getattr__(self, item):
        if item in self.keys():
            return self[item]
        elif item in self.__dict__:
            return self.__dict__[item]
        else:
            raise KeyError('no item named: {}'.format(item))

    def __setattr__(self, key, value):
        self[key] = value

    def __hasattr__(self, key):
        if key in self.keys() or key in self.__dict__:
            return True
        else:
            return False


def merge_a_to_b(dic: dict, attr_dict: AttrDict):
    assert isinstance(dic, dict)
    assert isinstance(attr_dict, AttrDict)

    for key in dic.keys():
        assert hasattr(attr_dict, key)
        if isinstance(dic[key], dict):
            merge_a_to_b(dic[key], attr_dict[key])
        else:
            assert type(dic[key]) == type(attr_dict[key]), "attr_dict['{}'] type: {}, dict['{}'] type: {}".format(key, type(attr_dict[key]),
                                                                                                                  key,
                                                                                                                  type(dic[key]))
            attr_dict[key] = dic[key]
