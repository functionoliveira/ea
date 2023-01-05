import datetime as dt
import neuroevo.messages as messages
from neuroevo.shadeils import ShadeILS
from neuroevo.solution import SingleLayerSolution
from neuroevo.settings import Settings
from utils.general import raise_if, pad_left


class DownShadeILS(ShadeILS):
    """Success-History Based Parameter Adaptation for Differential Evolution

    Args:
        EA (Object): Classe base para algoritmos de evolução
    """
    def __init__(self, settings):
        raise_if(not isinstance(settings, Settings), messages.POPULATION_TOO_SMALL, TypeError)
        raise_if(not isinstance(settings.solution, SingleLayerSolution), messages.SOLUTION_VALUE_ERROR, ValueError)
        super().__init__(settings.solution, settings.maxevals, settings.threshold, settings.generation, settings.popsize, settings.debug, settings.log, settings.identity)
        self.epochs = settings.epochs
        self.root_folder = f'output/down_shadeils/{settings.identity}' if settings.identity is not None else f'output/down_shadeils/{dt.datetime.now().strftime("%Y%m%d")}'
        self.log = settings.log(f'./{self.root_folder}')
        self.compound_folder = ['', '']
        self.dataloader = settings.dataloader
        self.dataset = settings.dataset

    def get_layers(self):
        return [idx for idx, _ in enumerate(self.solution.model.parameters())]

    def evolve(self):
        self.log.info(f"Starting algorithm={self.__class__.__name__} popsize={self.popsize} generations={len(self.G)} max_evals={self.maxevals} epochs={self.epochs} threshold={self.threshold} mutation={self.mutation_method.__class__.__name__}")
        EPOCHS = [e for e in range(self.epochs)]
        layers = self.get_layers()
        e_padding = len(list(str(self.epochs)))
        l_padding = len(list(str(len(layers))))

        for e in EPOCHS:
            self.compound_folder[0] = f'epoch_{pad_left(e, e_padding)}'
            for batch, data in enumerate(self.dataloader):
                input, output = data
                for l in layers:
                    self.compound_folder[1] = f'layer_{pad_left(l, l_padding)}'
                    self.solution.set_data(input, output)
                    self.population = self.solution.reload_chromosome(l, self.population)
                    super().evolve()
                    self.log.info(f"algorithm={self.__class__.__name__} epoch={e} batch={batch} layer={l} best_fitness={self.best_global_fitness}")
                input, output = self.dataset[:]
                loss = self.solution.fitness(self.best_global)
                self.log.info(f"algorithm={self.__class__.__name__} epoch={e} batch={batch} loss={loss}")
            self.log.info(f"algorithm={self.__class__.__name__} epoch={e} best_fitness={self.best_global_fitness}")

class UpShadeILS(DownShadeILS):
    def __init__(self, settings):
        super().__init__(settings)
        self.root_folder = f'output/up_shadeils/{settings.identity}' if settings.identity is not None else f'output/up_shadeils/{dt.datetime.now().strftime("%Y%m%d")}'
        self.log = settings.log(f'./{self.root_folder}')
    
    def get_layers(self):
        layers = [idx for idx, _ in enumerate(self.solution.model.parameters())]
        layers.reverse()
        return layers

    def evolve(self):
        super().evolve()
