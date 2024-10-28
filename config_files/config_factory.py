from config_files.Epilepsy_Configs import Config as EpilepsyConfig
from config_files.HAR_Configs import Config as HARConfig
from config_files.pFD_Configs import Config as PFDConfig
from config_files.sleepEDF_Configs import Config as SleepEDFConfig


class ConfigFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, name, builder):
        self._builders[name] = builder

    def create(self, name):
        builder = self._builders.get(name)
        if not builder:
            raise ValueError('Unknown builder {}'.format(name))
        return builder()

config_factory = ConfigFactory()
config_factory.register_builder('Epilepsy', EpilepsyConfig)
config_factory.register_builder('HAR', HARConfig)
config_factory.register_builder('pFD', PFDConfig)
config_factory.register_builder('EEG', SleepEDFConfig)
