import os
import sys
import numpy as np
import pandas as pd

from pywr.parameters import load_parameter, Parameter
from pywr.recorders import (NumpyArrayNodeRecorder, NodeRecorder, Aggregator, NumpyArrayStorageRecorder, NumpyArrayAbstractStorageRecorder, 
                            Recorder, hydropower_calculation, NumpyArrayParameterRecorder, BaseConstantParameterRecorder)
from pywr.recorders._recorders import NumpyArrayNodeRecorder

class NumpyArrayAnnualNodeDeficitFrequencyRecorder(NodeRecorder):

    """
    Number of two consecutive years where there is a deficit greater that a threshold
    """

    def __init__(self, model, node, threshold, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'COUNT_NONZERO')
        
        super().__init__(model, node, **kwargs)
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self.threshold = threshold

        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)

            if node.flow[scenario_index.global_id] < max_flow * self.threshold:
                
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def to_dataframe(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(np.array(self._data), index=index, columns=sc_index)


    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        count_nonzeros = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('Y').sum().to_numpy()

        return self._temporal_aggregator.aggregate_2d(count_nonzeros, axis=0, ignore_nan=self.ignore_nan)

    def aggregated_value(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        annual_val = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('Y').sum().to_numpy()

        zeros_ones = np.where(annual_val > 0, 1, 0)

        tuples = []
        for i in range(0, len(zeros_ones), 1):
            if i == 0:
                continue
            else:
                tem = [zeros_ones[i-1], zeros_ones[i]]
                tuples.append(tem)

        count = sum([all(np.array(x) == 1) for x in tuples])

        return count


NumpyArrayAnnualNodeDeficitFrequencyRecorder.register()


class AbstractComparisonNodeRecorder(NumpyArrayNodeRecorder):
    """ Base class for all Recorders performing timeseries comparison of `Node` flows
    """

    # def __init__(self, model, node, observed, obs_freq=None, **kwargs):
    #     super(AbstractComparisonNodeRecorder, self).__init__(model, node, **kwargs)

    def __init__(self, model, node, observed, obs_freq=None, **kwargs):
        NumpyArrayNodeRecorder.__init__(self, model, node, **kwargs)

        self.observed = observed
        self._aligned_observed = None
        self.obs_freq = obs_freq

    def setup(self):
        # super(AbstractComparisonNodeRecorder, self).setup()
        NumpyArrayNodeRecorder.setup(self)
        # Align the observed data to the model

        from pywr.parameters import align_and_resample_dataframe

        freq = self.obs_freq
        index_col = self.observed.index.tolist()
        start, end = index_col[0], index_col[-1]
        timestepper = pd.period_range(start=start, end=end, freq=freq)
        self._aligned_observed = align_and_resample_dataframe(self.observed, timestepper, 'sum')
        #self._aligned_observed = align_and_resample_dataframe(self.observed, self.model.timestepper.datetime_index)

    @classmethod
    def load(cls, model, data):
        # called when the parameter is loaded from a JSON document

        observed = data.pop("observed")
        index_col = data.pop("index_col")
        url = data.pop("url")
        obs_freq = data.pop("obs_freq")

        if '.csv' in url:
            data_observed = pd.read_csv(url)

        if 'xlsx' in url:
            data_observed = pd.read_excel(url)

        observed = pd.DataFrame(data=np.array(data_observed[observed]), index=data_observed[index_col])

        node = model._get_node_from_ref(model, data.pop("node"))

        return cls(model, node, observed, obs_freq, **data)


class RootMeanSquaredErrorNodeRecorder(AbstractComparisonNodeRecorder):
    """ Recorder evaluates the RMSE between model and observed """
    def values(self):

        freq = self.obs_freq
        obs = self._aligned_observed
        mod = self.data

        if freq is None:
            mod = self.data
        else:
            if self.model.timestepper.freq == freq:
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).mean()
            else:
                #print(f'OJO! The recorder associated to this node "{self.node.name}" '
                      #f'has freq observed data =! freq model - '
                      #f'Check if freq observed data >= freq model.')
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index.astype('datetime64[ns]')).resample(freq).mean()
                mod.index = mod.index.strftime('%Y-%m')
                obs.index = obs.index.astype('datetime64[ns]').strftime('%Y-%m')

            #mod = pandas.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).sum()

        new = pd.merge(obs, mod, how='inner', left_index=True, right_index=True)
        obs = new.iloc[:, 0].to_frame().T.reset_index(drop=True).T
        mod = new.iloc[:, 1].to_frame().T.reset_index(drop=True).T

        val = np.sqrt(np.mean((obs - mod) ** 2, axis=0))

        return val.values


RootMeanSquaredErrorNodeRecorder.register()


class NashSutcliffeEfficiencyNodeRecorder(AbstractComparisonNodeRecorder):
    """ Recorder evaluates the Nash-Sutcliffe efficiency model and observed """
    def values(self):

        freq = self.obs_freq
        obs = self._aligned_observed

        mod = self.data

        if freq is None:
            mod = self.data
        else:
            if self.model.timestepper.freq == freq:
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).mean()
            else:
                #print(f'OJO! The recorder associated to this node "{self.node.name}" '
                      #f'has freq observed data =! freq model - '
                      #f'Check if freq observed data >= freq model.')
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index.astype('datetime64[ns]')).resample(freq).mean()
                mod.index = mod.index.strftime('%Y-%m')
                obs.index = obs.index.astype('datetime64[ns]').strftime('%Y-%m')

            #mod = pandas.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).sum()

        new = pd.merge(obs,mod, how='inner', left_index=True, right_index=True)
        obs = new.iloc[:, 0].to_frame().T.reset_index(drop=True).T
        mod = new.iloc[:, 1].to_frame().T.reset_index(drop=True).T

        obs_mean = np.mean(obs, axis=0)

        val = 1.0 - np.sum((obs-mod)**2, axis=0)/np.sum((obs-obs_mean)**2, axis=0)

        return val.values


NashSutcliffeEfficiencyNodeRecorder.register()


class PercentBiasNodeRecorder(AbstractComparisonNodeRecorder):
    """ Recorder evaluates the absolute percent bias between model and observed """
    def values(self):

        freq = self.obs_freq
        obs = self._aligned_observed
        mod = self.data

        if freq is None:
            mod = self.data
        else:
            if self.model.timestepper.freq == freq:
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).mean()
            else:
                #print(f'OJO! The recorder associated to this node "{self.node.name}" '
                      #f'has freq observed data =! freq model - '
                      #f'Check if freq observed data >= freq model.')
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index.astype('datetime64[ns]')).resample(freq).mean()
                mod.index = mod.index.strftime('%Y-%m')
                obs.index = obs.index.astype('datetime64[ns]').strftime('%Y-%m')

            #mod = pandas.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).sum()

        new = pd.merge(obs,mod, how='inner', left_index=True, right_index=True)
        obs = new.iloc[:, 0].to_frame().T.reset_index(drop=True).T
        mod = new.iloc[:, 1].to_frame().T.reset_index(drop=True).T

        # val = np.sum(obs-mod, axis=0)*100/np.sum(obs, axis=0)
        val = np.abs(np.sum(obs - mod, axis=0) * 100 / np.sum(obs, axis=0))

        return val.values


PercentBiasNodeRecorder.register()


class AbstractComparisonStorageRecorder(NumpyArrayStorageRecorder):
    """ Base class for all Recorders performing timeseries comparison of `Storage Node`
    """

    def __init__(self, model, node, observed, obs_freq=None, **kwargs):
        super(AbstractComparisonStorageRecorder, self).__init__(model, node, **kwargs)

        self.observed = observed
        self._aligned_observed = None
        self.obs_freq = obs_freq

    def setup(self):
        super(AbstractComparisonStorageRecorder, self).setup()
        # Align the observed data to the model

        from pywr.parameters import align_and_resample_dataframe
        #        self._aligned_observed = align_and_resample_dataframe(self.observed, self.model.timestepper.datetime_index)

        freq = self.obs_freq
        index_col = self.observed.index.tolist()
        start, end = index_col[0], index_col[-1]
        timestepper = pd.period_range(start=start, end=end, freq=freq)
        self._aligned_observed = align_and_resample_dataframe(self.observed, timestepper, 'mean')

    @classmethod
    def load(cls, model, data):
        # called when the parameter is loaded from a JSON document

        observed = data.pop("observed")
        index_col = data.pop("index_col")
        url = data.pop("url")
        obs_freq = data.pop("obs_freq")

        if '.csv' in url:
            data_observed = pd.read_csv(url)

        if 'xlsx' in url:
            data_observed = pd.read_excel(url)

        observed = pd.DataFrame(data=np.array(data_observed[observed]), index=data_observed[index_col])

        node = model._get_node_from_ref(model, data.pop("node"))

        return cls(model, node, observed, obs_freq, **data)


class NashSutcliffeEfficiencyStorageRecorder(AbstractComparisonStorageRecorder):
    """ Recorder evaluates the Nash-Sutcliffe efficiency model and observed """

    def values(self):

        freq = self.obs_freq
        obs = self._aligned_observed

        mod = self.data

        if freq is None:
            mod = self.data
        else:
            if self.model.timestepper.freq == freq:
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).mean()
            else:
                #print(f'OJO! The recorder associated to this node "{self.node.name}" '
                      #f'has freq observed data =! freq model - '
                      #f'Check if freq observed data >= freq model')
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index.astype('datetime64[ns]')).resample(freq).mean()
                mod.index = mod.index.strftime('%Y-%m')
                obs.index = obs.index.astype('datetime64[ns]').strftime('%Y-%m')

        new = pd.merge(obs, mod, how='inner', left_index=True, right_index=True)
        obs = new.iloc[:, 0].to_frame().T.reset_index(drop=True).T
        mod = new.iloc[:, 1].to_frame().T.reset_index(drop=True).T

        obs_mean = np.mean(obs, axis=0)

        val = 1.0 - np.sum((obs - mod) ** 2, axis=0) / np.sum((obs - obs_mean) ** 2, axis=0)

        return val.values


NashSutcliffeEfficiencyStorageRecorder.register()


class RootMeanSquaredErrorStorageRecorder(AbstractComparisonStorageRecorder):
    """ Recorder evaluates the RMSE between model and observed """
    def values(self):

        freq = self.obs_freq
        obs = self._aligned_observed
        mod = self.data

        if freq is None:
            mod = self.data
        else:
            if self.model.timestepper.freq == freq:
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).mean()
            else:
                #print(f'OJO! The recorder associated to this node "{self.node.name}" '
                      #f'has freq observed data =! freq model - '
                      #f'Check if freq observed data >= freq model.')
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index.astype('datetime64[ns]')).resample(freq).mean()
                mod.index = mod.index.strftime('%Y-%m')
                obs.index = obs.index.astype('datetime64[ns]').strftime('%Y-%m')

            #mod = pandas.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).sum()

        new = pd.merge(obs, mod, how='inner', left_index=True, right_index=True)
        obs = new.iloc[:, 0].to_frame().T.reset_index(drop=True).T
        mod = new.iloc[:, 1].to_frame().T.reset_index(drop=True).T

        val = np.sqrt(np.mean((obs - mod) ** 2, axis=0))

        return val.values


RootMeanSquaredErrorStorageRecorder.register()


class PercentBiasStorageRecorder(AbstractComparisonStorageRecorder):
    """ Recorder evaluates the percent bias between model and observed """
    def values(self):

        freq = self.obs_freq
        obs = self._aligned_observed
        mod = self.data

        if freq is None:
            mod = self.data
        else:
            if self.model.timestepper.freq == freq:
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).mean()
            else:
                #print(f'OJO! The recorder associated to this node "{self.node.name}" '
                      #f'has freq observed data =! freq model - '
                      #f'Check if freq observed data >= freq model.')
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index.astype('datetime64[ns]')).resample(freq).mean()
                mod.index = mod.index.strftime('%Y-%m')
                obs.index = obs.index.astype('datetime64[ns]').strftime('%Y-%m')

            #mod = pandas.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).sum()

        new = pd.merge(obs,mod, how='inner', left_index=True, right_index=True)
        obs = new.iloc[:, 0].to_frame().T.reset_index(drop=True).T
        mod = new.iloc[:, 1].to_frame().T.reset_index(drop=True).T

        val = np.sum(obs-mod, axis=0)*100/np.sum(obs, axis=0)

        return val.values


PercentBiasStorageRecorder.register()


class ReservoirMonthlyReliabilityRecorder(NumpyArrayAbstractStorageRecorder):

    """
    1 - (Total months below minimum storage level / total months in simulation)
    """

    def __init__(self, model, node, threshold, **kwargs):
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_volume = node.get_max_volume(scenario_index)

            if node.volume[scenario_index.global_id] < max_volume * self.threshold:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        DataFrame = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('M').max()

        return 1 - ((DataFrame.sum().round(0) / DataFrame.shape[0]))
    
    def to_dataframe(self):

        raise NotImplementedError()


ReservoirMonthlyReliabilityRecorder.register()


class ReservoirAnnualReliabilityRecorder(NumpyArrayAbstractStorageRecorder):

    """
    1 - (Total years below minimum storage level / total years in simulation)
    """

    def __init__(self, model, node, threshold, **kwargs):
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_volume = node.get_max_volume(scenario_index)

            if node.volume[scenario_index.global_id] < max_volume * self.threshold:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        DataFrame = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('Y').max()

        return 1 - ((DataFrame.sum().round(0) / DataFrame.shape[0]))
    
    def to_dataframe(self):
        
        raise NotImplementedError()


ReservoirAnnualReliabilityRecorder.register()


class SupplyReliabilityRecorder(NodeRecorder):

    """
    add description
    """

    def __init__(self, model, node, **kwargs):
        super().__init__(model, node, **kwargs)

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)

            if max_flow == 0:
                deficit = 0

            else:
                deficit  = (max_flow - node.flow[scenario_index.global_id]) / max_flow

            if deficit > 0.01:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        DataFrame = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('M').max().loc[:str(last_year), :]

        return 1 - ((DataFrame.sum().round(0) / DataFrame.shape[0]))
    
    def to_dataframe(self):
        
        raise NotImplementedError()


SupplyReliabilityRecorder.register()


class AnnualDeficitRecorder(NodeRecorder):

    """
    Annual deficit recorder (%)
    """

    def __init__(self, model, node, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        
        super().__init__(model, node, **kwargs)
        self.temporal_aggregator = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]

        rlts = 1 - supply.divide(demand)

        if self.temporal_aggregator == 'mean':
            to_save = rlts.mean()

        if self.temporal_aggregator == 'max':
            to_save = rlts.max()

        if self.temporal_aggregator == 'min':
            to_save = rlts.min()

        return to_save
    
    def to_dataframe(self):
        
        raise NotImplementedError()


AnnualDeficitRecorder.register()


class ReservoirResilienceRecorder(NumpyArrayAbstractStorageRecorder):

    """
    add description
    """

    def __init__(self, model, node, threshold, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        
        super().__init__(model, node, **kwargs)
        self.temporal_aggregator = temporal_agg_func
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_volume = node.get_max_volume(scenario_index)

            if node.volume[scenario_index.global_id] < max_volume * self.threshold:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        tem_dams = pd.DataFrame(np.array(self._data), index=index, columns=sc_index)

        tem_dams_diff = tem_dams.diff().ne(0).cumsum()
        
        tem_dams_occurrence = tem_dams.multiply(tem_dams_diff)

        resilience = {}
        
        levels = [x for x, _ in enumerate(tem_dams_occurrence.columns.names)]
        
        for idx, dataframe in tem_dams_occurrence.groupby(level=levels, axis=1):
            
            tem = dataframe.T.reset_index(drop=True).T
            
            tem.columns = ['col']
            
            tem_res = tem[tem['col'] != 0].groupby(['col'])['col'].count()

            if self.temporal_aggregator == 'mean':
                resilience[idx] = tem_res.mean()
                
            if self.temporal_aggregator == 'max':
                resilience[idx] = tem_res.max()
            
        
        rlts = pd.DataFrame.from_dict(resilience, orient='index', columns=[""])

        rlts = rlts.T

        rlts.columns = pd.MultiIndex.from_tuples(rlts.columns, names=sc_index.names)

        return rlts.T
    
    def to_dataframe(self):
        
        raise NotImplementedError()


ReservoirResilienceRecorder.register()


class RelativeCropYieldRecorder(Recorder):
    """Relative crop yield recorder.

    This recorder computes the relative crop yield based on a curtailment ratio between a node's
    actual flow and it's `max_flow` expected flow. It is assumed the `max_flow` parameter is an
    `AggregatedParameter` containing only `IrrigationWaterRequirementParameter` parameters.

    """
    def __init__(self, model, nodes, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        super().__init__(model, **kwargs)

        for node in nodes:
            max_flow_param = node.max_flow
            self.children.add(max_flow_param)

        self.nodes = nodes
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self.data = None

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        self.data = np.zeros((nts, ncomb))

    def reset(self):
        self.data[:, :] = 0.0

    def after(self):

        norm_crop_revenue = None
        full_norm_crop_revenue = None
        ts = self.model.timestepper.current
        self.data[ts.index, :] = 0
        norm_yield = 0
        full_norm_yield = 0

        for node in self.nodes:
            crop_aggregated_parameter = node.max_flow
            actual = node.flow
            requirement = np.array(crop_aggregated_parameter.get_all_values())
            # Divide non-zero elements
            curtailment_ratio = np.divide(actual, requirement, out=np.zeros_like(actual), where=requirement != 0)
            no_curtailment = np.ones_like(curtailment_ratio)

            if norm_crop_revenue is None:
                norm_crop_revenue = crop_aggregated_parameter.parameters[0].crop_revenue(curtailment_ratio)
                full_norm_crop_revenue = crop_aggregated_parameter.parameters[0].crop_revenue(no_curtailment)

            for parameter in crop_aggregated_parameter.parameters:
                crop_revenue = parameter.crop_revenue(curtailment_ratio)
                full_crop_revenue = parameter.crop_revenue(no_curtailment)
                crop_yield = parameter.crop_yield(curtailment_ratio)
                full_crop_yield = parameter.crop_yield(no_curtailment)
                # Increment effective yield, scaled by the first crop's revenue
                norm_yield += crop_yield * np.divide(crop_revenue, norm_crop_revenue,
                                                    out=np.zeros_like(crop_revenue),
                                                    where=norm_crop_revenue != 0)

                full_norm_yield += full_crop_yield * np.divide(full_crop_revenue, full_norm_crop_revenue,
                                                              out=np.ones_like(full_crop_revenue),
                                                              where=full_norm_crop_revenue != 0)
                
                if requirement<0.00001:
                    self.data[ts.index, :] = 99999
                else:
                    self.data[ts.index, :] = norm_yield / full_norm_yield

    def values(self):
        """Compute a value for each scenario using `temporal_agg_func`.
        """
        return self._temporal_aggregator.aggregate_2d(self.data, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self.data), index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        nodes = [model._get_node_from_ref(model, n) for n in data.pop('nodes')]
        return cls(model, nodes, **data)

RelativeCropYieldRecorder.register()


class AverageAnnualCropYieldScenarioRecorder(NodeRecorder):

    """
    This recorder computes the average annual crop yield for each scenario based on the curtailment ratio between a node's
    """

    def __init__(self, model, node, threshold=None, **kwargs):

        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs)
        self.threshold = threshold
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))
        self._yield = np.zeros((nts, ncomb))
        self._area = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
        self._yield[:, :] = 0.0
        self._area[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)
    
            if isinstance(node.max_flow.yield_per_area, float):
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area
            else:
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area.get_value(scenario_index)

            if isinstance(node.max_flow.area, float):
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area
            else:
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area.get_value(scenario_index)

        return 0

    def to_dataframe(self):
    
        max_flow_param = self.node.max_flow
        
        index = self.model.timestepper.datetime_index
        resample_index = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=resample_index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=resample_index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        areas = pd.DataFrame(np.array(self._area), index=resample_index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]
        yields = pd.DataFrame(np.array(self._yield), index=resample_index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        # units for yields are in kg/ha
        # units for areas are in ha
        # units for crop_yield are in kg
        
        return curtailment_ratio.multiply(areas).multiply(yields)


    def values(self):
        
        max_flow_param = self.node.max_flow
        
        index = self.model.timestepper.datetime_index
        resample_index = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=resample_index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=resample_index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        areas = pd.DataFrame(np.array(self._area), index=resample_index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]
        yields = pd.DataFrame(np.array(self._yield), index=resample_index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        # units for yields are in kg/ha
        # units for areas are in ha
        # units for crop_yield are in kg
        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(yields, axis=1)

        return self._temporal_aggregator.aggregate_2d(crop_yield.values, axis=0, ignore_nan=self.ignore_nan) #crop_yield.mean(axis=0) 


AverageAnnualCropYieldScenarioRecorder.register()


class TotalAnnualCropYieldScenarioRecorder(NodeRecorder):

    """
    This recorder computes the Total annual crop yield for each scenario assuming there is anough water to irrigate the crop
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))
        self._yield = np.zeros((nts, ncomb))
        self._area = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
        self._yield[:, :] = 0.0
        self._area[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

            if isinstance(node.max_flow.yield_per_area, float):
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area
            else:
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area.get_value(scenario_index)

            if isinstance(node.max_flow.area, float):
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area
            else:
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area.get_value(scenario_index)

        return 0

    def to_dataframe(self):
        
        raise NotImplementedError()


    def values(self):
        
        max_flow_param = self.node.max_flow
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        #supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        areas = pd.DataFrame(np.array(self._area), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]
        yields = pd.DataFrame(np.array(self._yield), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]

        curtailment_ratio = demand.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(yields, axis=1)

        return crop_yield.mean(axis=0)


TotalAnnualCropYieldScenarioRecorder.register()


class IrrigationSupplyReliabilityScenarioRecorder(NodeRecorder):

    """
    This recorder calculates the supply reliability of an irrigation node considering only the months with higher demand 
    based on the Kc parameter. 
    
    A month with high demand > 0.8 Kc
    A year is considered that fails if the supply is less than 80% of the demand in any of the months with higher demand
    The supply reliability is calculated as (1 - ((number of years of failure / number of years in the simulation)))
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))
        self._kc = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
        self._kc[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node
        max_flow_param = self.node.max_flow

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)
            self._kc[ts.index, scenario_index.global_id] = max_flow_param.crop_water_factor_parameter.get_value(scenario_index)

        return 0

    def to_dataframe(self):
        
        raise NotImplementedError()


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('M').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('M').sum().loc[:str(last_year), :]

        # Here we calculate the months where the demand is higher than 0.8 Kc
        mths_kc = np.where(self._kc < np.max(self._kc)*0.8, 0, 1)
        mths_kc = pd.DataFrame(mths_kc, index=index, columns=sc_index).resample('M').mean().loc[:str(last_year), :]

        moths_failures = np.where(supply < demand*self.threshold, 1, 0)
        moths_failures = pd.DataFrame(moths_failures, index=demand.index, columns=demand.columns)

        # Here we calculate the years where there is a failure only considering the months with high demand "mths_kc"
        failures = moths_failures.multiply(mths_kc).dropna().resample('Y').max()


        return 1 - (failures.sum().round(0) / failures.shape[0])


IrrigationSupplyReliabilityScenarioRecorder.register()


class CropCurtailmentRatioScenarioRecorder(NodeRecorder):

    """
    This recorder save the Annual Curtailment Ratios
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

        return 0

    def to_dataframe(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        
        return curtailment_ratio


    def values(self):
        

        return NotImplementedError()


CropCurtailmentRatioScenarioRecorder.register()


class AnnualIrrigationSupplyReliabilityScenarioRecorder(NodeRecorder):

    """
    This recorder calculates the annual supply reliability based on a threashold. 

    A year is considered that fails if the supply is less than a threashold
    The supply reliability is calculated as (1 - ((number of years of failure / number of years in the simulation)))
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

        return 0

    def to_dataframe(self):
        
        raise NotImplementedError()


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]


        # Here we calculate the years where there is a failure only considering the threshold
        failures = np.where(supply < demand*self.threshold, 1, 0)
        failures = pd.DataFrame(failures, index=demand.index, columns=demand.columns)

        return 1 - (failures.sum().round(0) / failures.shape[0])


AnnualIrrigationSupplyReliabilityScenarioRecorder.register()


class AverageAnnualIrrigationRevenueScenarioRecorder(NodeRecorder):

    """
    This recorder computes the average annual irrigation revenue for each scenario based on the curtailment ratio between a node's
    the price should be imput in $/tn
    """

    def __init__(self, model, node, threshold=None, price=1, **kwargs):

        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs)
        self.threshold = threshold
        self.price = price
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))
        self._yield = np.zeros((nts, ncomb))
        self._area = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
        self._yield[:, :] = 0.0
        self._area[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

            if isinstance(node.max_flow.yield_per_area, float):
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area
            else:
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area.get_value(scenario_index)

            if isinstance(node.max_flow.area, float):
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area
            else:
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area.get_value(scenario_index)

        return 0

    def to_dataframe(self):

        max_flow_param = self.node.max_flow
        
        index = self.model.timestepper.datetime_index
        resample_index = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=resample_index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=resample_index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        areas = pd.DataFrame(np.array(self._area), index=resample_index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]
        yields = pd.DataFrame(np.array(self._yield), index=resample_index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        # units for yields are in kg/ha
        # units for areas are in ha
        # units for crop_yield are in kg
        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(yields, axis=1)
        
        return crop_yield.divide(1e3).multiply(self.price).divide(1e6) # kg to tn then $ to M$


    def values(self):

        max_flow_param = self.node.max_flow
        
        index = self.model.timestepper.datetime_index
        resample_index = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=resample_index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=resample_index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        areas = pd.DataFrame(np.array(self._area), index=resample_index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]
        yields = pd.DataFrame(np.array(self._yield), index=resample_index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0
        curtailment_ratio.fillna(0, inplace=True)

        # units for yields are in kg/ha
        # units for areas are in ha
        # units for crop_yield are in kg
        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(yields, axis=1)
        revenue = crop_yield.divide(1e3).multiply(self.price).divide(1e6) # kg to tn then $ to M$

        return self._temporal_aggregator.aggregate_2d(revenue.dropna().values, axis=0, ignore_nan=self.ignore_nan)
    

AverageAnnualIrrigationRevenueScenarioRecorder.register()
    
                
class AnnualSeasonalAccumulatedFlowRecorder(NodeRecorder):

    """
    This recorder calculates the total annual flow using the months defined per user. 
    It counts from the 1st of the first month to the last day of the last month.
    """

    def __init__(self, model, node, months=None, **kwargs):
        
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs)
        self.months = months
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self.cummulatedFlow = np.zeros((nts, ncomb))

    def reset(self):
        self.cummulatedFlow[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self.cummulatedFlow[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]

        return 0

    def to_dataframe(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        AnnualFlow = pd.DataFrame(np.array(self.cummulatedFlow), index=index, columns=sc_index)
        AnnualFlow = AnnualFlow.resample('D').ffill()

        AnnualFlow = AnnualFlow[AnnualFlow.index.month.isin(self.months)]

        last_year = index[-1].year
        AnnualFlow = AnnualFlow.loc[:str(last_year), :].resample('Y').sum()
        
        return AnnualFlow


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        AnnualFlow = pd.DataFrame(np.array(self.cummulatedFlow), index=index, columns=sc_index)
        AnnualFlow = AnnualFlow.resample('D').ffill()
        AnnualFlow = AnnualFlow[AnnualFlow.index.month.isin(self.months)]

        last_year = index[-1].year
        AnnualFlow = AnnualFlow.loc[:str(last_year), :].resample('Y').sum()

        return self._temporal_aggregator.aggregate_2d(AnnualFlow.values, axis=0, ignore_nan=self.ignore_nan)


AnnualSeasonalAccumulatedFlowRecorder.register()


class AnnualSeasonalVolumeRecorder(NodeRecorder):

    """
    This recorder calculates the total annual flow using the months defined per user. 
    It counts from the 1st of the first month to the last day of the last month.
    """

    def __init__(self, model, node, months=None, **kwargs):
        
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs)
        self.months = months
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self.cummulatedFlow = np.zeros((nts, ncomb))

    def reset(self):
        self.cummulatedFlow[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self.cummulatedFlow[ts.index, scenario_index.global_id] = node.volume[scenario_index.global_id]

        return 0

    def to_dataframe(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        AnnualVolume = pd.DataFrame(np.array(self.cummulatedFlow), index=index, columns=sc_index)
        AnnualVolume = AnnualVolume.resample('D').ffill()
        AnnualVolume = AnnualVolume[AnnualVolume.index.month.isin(self.months)]

        last_year = index[-1].year
        AnnualVolume = AnnualVolume.loc[:str(last_year), :].resample('Y').mean()
        
        return AnnualVolume


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        AnnualVolume = pd.DataFrame(np.array(self.cummulatedFlow), index=index, columns=sc_index)
        AnnualVolume = AnnualVolume.resample('D').ffill()
        AnnualVolume = AnnualVolume[AnnualVolume.index.month.isin(self.months)]

        last_year = index[-1].year
        AnnualVolume = AnnualVolume.loc[:str(last_year), :].resample('Y').mean()
        
        #AnnualVolume = AnnualVolume.mean(axis=0)
        
        # get the value only
        #AnnualVolume = AnnualVolume.values

        return self._temporal_aggregator.aggregate_2d(AnnualVolume.values, axis=0, ignore_nan=self.ignore_nan) 


AnnualSeasonalVolumeRecorder.register()

class AnnualHydropowerRecorder(NumpyArrayNodeRecorder):
    """ Calculates the annual power production using the hydropower equation
    This recorder is inspired on the `pywr.recorders.HydropowerRecorder` but it is
    designed to be used with the `pywr.recorders.NumpyArrayNodeRecorder` class. 

    Parameters
    ----------

    water_elevation_parameter : Parameter instance (default=None)
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
    turbine_elevation : double
        Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
        the working head of the turbine.
    efficiency : float (default=1.0)
        The efficiency of the turbine.
    density : float (default=1000.0)
        The density of water.
    flow_unit_conversion : float (default=1.0)
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of :math:`m^3/day`
    energy_unit_conversion : float (default=1e-6)
        A factor used to transform the units of total energy. Defaults to 1e-6 to return :math:`MJ`.

    Notes
    -----
    The hydropower calculation uses the following equation.

    .. math:: P = \\rho * g * \\delta H * q

    The flow rate in should be converted to units of :math:`m^3` per day using the `flow_unit_conversion` parameter.

    Head is calculated from the given `water_elevation_parameter` and `turbine_elevation` value. If water elevation
    is given then head is the difference in elevation between the water and the turbine. If water elevation parameter
    is `None` then the head is simply the turbine elevation.


    See Also
    --------
    TotalHydroEnergyRecorder
    pywr.parameters.HydropowerTargetParameter

    """
    def __init__(self, model, node, monthly_seasonality, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000.0,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs) # Changed from super(HydropowerRecorder, self) for Python 3+ style

        # Initialize _water_elevation_parameter before setting the property
        # to ensure the setter can access it if it checks hasattr(self, '_water_elevation_parameter').
        # However, direct assignment to the property will call the setter.
        self._water_elevation_parameter = None 
        self.water_elevation_parameter = water_elevation_parameter # Use the setter

        self._monthly_seasonality = monthly_seasonality
        self.turbine_elevation = float(turbine_elevation)
        self.efficiency = float(efficiency)
        self.density = float(density)
        self.flow_unit_conversion = float(flow_unit_conversion)
        self.energy_unit_conversion = float(energy_unit_conversion)

        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    @property
    def water_elevation_parameter(self):
        """The water elevation parameter instance."""
        return self._water_elevation_parameter

    @water_elevation_parameter.setter
    def water_elevation_parameter(self, parameter):
        """Sets the water elevation parameter, updating children."""
        current_parameter = getattr(self, '_water_elevation_parameter', None)

        if current_parameter: # If current_parameter is not None and truthy
            if current_parameter in self.children:
                self.children.remove(current_parameter)
        
        # The original Cython code `self.children.add(parameter)` would add the parameter
        # to the set `self.children` even if `parameter` is None.
        # This behavior is preserved here.
        self.children.add(parameter)
        
        self._water_elevation_parameter = parameter

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0

    def after(self):
        """Called after each timestep to record hydropower production."""
        # Type hints for clarity, not strict enforcement in standard Python
        # q: float
        # head: float
        # power: float
        
        ts = self.model.timestepper.current
        # scenario_index: ScenarioIndex # Type hint for loop variable

        # Assuming self.node is set by the parent class (NumpyArrayNodeRecorder or NodeRecorder)
        # If not, and self._node is used:
        # node_flow_attr = self._node.flow # Or however flow is accessed per scenario

        for scenario_index in self.model.scenarios.combinations:
            
            if self._water_elevation_parameter is not None:
                water_elev = self._water_elevation_parameter.get_value(scenario_index)
                head = water_elev - self.turbine_elevation
            else:
                # If water_elevation_parameter is None, head is taken as turbine_elevation.
                # This matches the docstring and the simplified logic from the Cython code
                # given turbine_elevation is always a float.
                head = self.turbine_elevation
            
            # Negative head is not physically valid for power generation
            head = max(head, 0.0)

            # Get the flow from the current node for the specific scenario
            # NodeRecorder (parent of NumpyArrayNodeRecorder) stores the node as self._node.
            # Flow for a scenario is typically accessed like this in Pywr:
            q = self.node.flow[scenario_index.global_id]

            # Calculate power using the external hydropower_calculation function
            power = hydropower_calculation(
                q, 
                head, 
                0.0,  # Assuming 0.0 for tailwater_elevation as in the Cython call
                self.efficiency, 
                density=self.density,
                flow_unit_conversion=self.flow_unit_conversion,
                energy_unit_conversion=self.energy_unit_conversion
            )
            
            # Store the calculated power
            # self._data is assumed to be a 2D NumPy array (timesteps, scenarios)
            self._data[ts.index, scenario_index.global_id] = power

    def values(self):
        """Compute a value for each scenario using `temporal_agg_func`.
        """

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        annual_hydropower = pd.DataFrame(np.array(self._data), index=index, columns=sc_index)

        annual_hydropower = annual_hydropower.resample('D').ffill()

        if self._monthly_seasonality is not None:
            annual_hydropower = annual_hydropower[annual_hydropower.index.month.isin(self._monthly_seasonality)]

        annual_hydropower = annual_hydropower.resample('Y').sum() # To get annual hydropower generation in MWh/year

        if self.factor is not None:
            annual_hydropower = annual_hydropower.multiply(self.factor, axis=0)
    
        return self._temporal_aggregator.aggregate_2d(annual_hydropower.values, axis=0, ignore_nan=self.ignore_nan) # Return revenues if a factor (price) is set
    

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data

        to_dataframe() is a method that returns the hydropower generation in energy units at annual scale.
        
        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        annual_hydropower = pd.DataFrame(np.array(self._data), index=index, columns=sc_index)

        annual_hydropower = annual_hydropower.resample('D').ffill()

        if self._monthly_seasonality is not None:
            annual_hydropower = annual_hydropower[annual_hydropower.index.month.isin(self._monthly_seasonality)]

        annual_hydropower = annual_hydropower.resample('Y').sum() # To get annual hydropower generation in MWh/year

        if self.factor is not None:
            annual_hydropower = annual_hydropower.multiply(self.factor, axis=0)

        return annual_hydropower

    @classmethod
    def load(cls, model, data):
        """Loads the recorder from a dictionary configuration."""
        # It's good practice to ensure 'pywr.parameters' is accessible
        # or handle potential ImportError if this code is part of a larger system.
        from pywr.parameters import load_parameter # Assuming pywr is structured this way

        node_name = data.pop("node")
        monthly_seasonality = data.pop("monthly_seasonality", None)
        node = model.nodes[node_name] # Get the actual node instance

        water_elevation_param_data = data.pop("water_elevation_parameter", None)
        water_elevation_parameter = None
        if water_elevation_param_data is not None:
            water_elevation_parameter = load_parameter(model, water_elevation_param_data)
        
        # Remaining items in data are passed as kwargs to __init__
        return cls(model, node, monthly_seasonality, water_elevation_parameter=water_elevation_parameter, **data)
    
    
AnnualHydropowerRecorder.register()

class SeasonalTransferConstraintRecorder(NodeRecorder):

    """
    This recorder calculates the total annual flow using the months defined per user. 
    It counts from the 1st of the first month to the last day of the last month.
    """

    def __init__(self, model, node, node_rule, monthly_seasonality=None, **kwargs):
        
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs)
        self.node_rule = node_rule
        self.monthly_seasonality = monthly_seasonality
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))
        self._data_rule = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
        self._data_rule[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node
        node_rule = self.node_rule

        for scenario_index in self.model.scenarios.combinations:

            self._data[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._data_rule[ts.index, scenario_index.global_id] = node_rule.flow[scenario_index.global_id]

        return 0

    def to_dataframe(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        outflow = pd.DataFrame(np.array(self._data), index=index, columns=sc_index)
        outflow = outflow.resample('D').ffill()

        outflow = outflow[outflow.index.month.isin(self.monthly_seasonality)]

        last_year = index[-1].year
        outflow = outflow.loc[:str(last_year), :].resample('Y').sum()

        rule = pd.DataFrame(np.array(self._data_rule), index=index, columns=sc_index)
        rule = rule.resample('D').ffill()

        rule = rule[rule.index.month.isin(self.monthly_seasonality)]

        last_year = index[-1].year
        rule = rule.loc[:str(last_year), :].resample('Y').sum()

        return (rule - 4200) - outflow


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        outflow = pd.DataFrame(np.array(self._data), index=index, columns=sc_index)
        outflow = outflow.resample('D').ffill()

        outflow = outflow[outflow.index.month.isin(self.monthly_seasonality)]

        last_year = index[-1].year
        outflow = outflow.loc[:str(last_year), :].resample('Y').sum()

        rule = pd.DataFrame(np.array(self._data_rule), index=index, columns=sc_index)
        rule = rule.resample('D').ffill()

        rule = rule[rule.index.month.isin(self.monthly_seasonality)]

        last_year = index[-1].year
        rule = rule.loc[:str(last_year), :].resample('Y').sum()

        constraint = (rule - 4200) - outflow

        return self._temporal_aggregator.aggregate_2d(constraint.values, axis=0, ignore_nan=self.ignore_nan)
    

    @classmethod
    def load(cls, model, data):
        """Loads the recorder from a dictionary configuration."""
        
        node_name = data.pop("node")
        node_rule_name = data.pop("node_rule", None)

        node = model.nodes[node_name] # Get the actual node instance
        node_rule = model.nodes[node_rule_name] # Get the actual node instance

        monthly_seasonality = data.pop("monthly_seasonality", None)
        
        # Remaining items in data are passed as kwargs to __init__
        return cls(model, node, node_rule, monthly_seasonality, **data)


SeasonalTransferConstraintRecorder.register()

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
import numpy as np
from pywr.parameters import load_parameter, Parameter


class BaseCostRecorder(Recorder):
    """
    Base class for CAPEX cost-curve recorders. Evaluates once per scenario.
    """
    def __init__(self, model, **kwargs):
        kwargs.setdefault('agg_func', 'mean')
        super().__init__(model, **kwargs)
        self.total_cost = None

    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        self.total_cost = np.zeros(ncomb, dtype=float)

    def reset(self):
        super().reset()
        if self.total_cost is not None:
            self.total_cost[...] = 0.0

    def values(self):
        return self.total_cost

    def aggregated_value(self):
        return self._scenario_aggregator.aggregate_1d(self.total_cost, ignore_nan=self.ignore_nan)


def _load_param_or_float(model, value):
    if isinstance(value, str):
        return load_parameter(model, value)
    elif isinstance(value, (int, float)):
        return value
    return value

def _get_val(param, scenario_index):
    if isinstance(param, Parameter):
        return float(param.get_value(scenario_index))
    return float(param)


class ConveyanceEfficiencyCostRecorder(BaseCostRecorder):
    """Recorder to calculate the CAPEX of improving canal conveyance efficiency.

    Calculates the total capital cost of improving canal conveyance efficiency for
    a given length of canal using a power-law relationship between the efficiency gain
    and the investment cost.

    cost = a * (max(0, efficiency - n0))^b * canal_length_km

    Attributes
    ----------
    losses_parameter : pywr.parameters.Parameter or float
        Provides the target conveyance losses as a fraction (where efficiency = 1 - losses).
    a : float
        The cost coefficient.
    b : float
        The exponent of the cost function.
    n0 : float
        The baseline conveyance efficiency (fraction).
    canal_length_km : pywr.parameters.Parameter or float
        The total length of the canal in kilometers.
    """

    def __init__(self, model, losses_parameter, a, b, n0, canal_length_km, **kwargs):
        super().__init__(model, **kwargs)
        self.losses_parameter = _load_param_or_float(model, losses_parameter)
        self.a = float(a)
        self.b = float(b)
        self.n0 = float(n0)
        self.canal_length_km = _load_param_or_float(model, canal_length_km)

        if isinstance(self.losses_parameter, Parameter):
            self.losses_parameter.parents.add(self)
        if isinstance(self.canal_length_km, Parameter):
            self.canal_length_km.parents.add(self)

    def after(self):
        """Evaluate the cost at the end of the simulation timestep."""
        ts = self.model.timestepper.current
        # We only need to calculate cost once, but `after` is called every timestep.
        # So we can calculate it on the first timestep and hold it, OR calculate it at the end.
        # Often static CAPEX parameters don't change over timesteps. Let's compute it at `finish`
        return 0

    def finish(self):
        """Calculate the fixed capital cost at the end of the simulation."""
        for scenario_index in self.model.scenarios.combinations:
            losses = _get_val(self.losses_parameter, scenario_index)
            efficiency = 1.0 - losses
            # Only account for positive gains above the baseline
            delta_efficiency = max(0.0, efficiency - self.n0)
            cost_per_km = self.a * (delta_efficiency ** self.b)
            
            canal_len = _get_val(self.canal_length_km, scenario_index)
            self.total_cost[scenario_index.global_id] = cost_per_km * canal_len


    @classmethod
    def load(cls, model, data):
        losses_parameter = data.pop("losses_parameter", data.pop("parameter", None))
        canal_length_km = data.pop("canal_length_km")
        a = data.pop("a")
        b = data.pop("b")
        n0 = data.pop("n0")
        return cls(model, losses_parameter, a, b, n0, canal_length_km, **data)

ConveyanceEfficiencyCostRecorder.register()


class ReservoirStorageExpansionCostRecorder(BaseCostRecorder):
    """Recorder to calculate the CAPEX of new reservoir storage.

    Calculates the capital cost of building new surface reservoir storage
    capacity using a power function that captures economies of scale.
    
    cost = a * (max(0, new_max_volume - current_max_volume))^b

    Attributes
    ----------
    new_max_volume : pywr.parameters.Parameter or float
        The proposed new storage capacity in MCM.
    current_max_volume : pywr.parameters.Parameter or float
        The current storage capacity in MCM.
    a : float
        The coefficient of the cost function.
    b : float
        The exponent of the cost function.
    """

    def __init__(self, model, new_max_volume, current_max_volume, a, b, **kwargs):
        super().__init__(model, **kwargs)
        self.new_max_volume = _load_param_or_float(model, new_max_volume)
        self.current_max_volume = _load_param_or_float(model, current_max_volume)
        self.a = float(a)
        self.b = float(b)

        if isinstance(self.new_max_volume, Parameter):
            self.new_max_volume.parents.add(self)
        if isinstance(self.current_max_volume, Parameter):
            self.current_max_volume.parents.add(self)

    def after(self):
        return 0

    def finish(self):
        for scenario_index in self.model.scenarios.combinations:
            v_new = _get_val(self.new_max_volume, scenario_index)
            v_cur = _get_val(self.current_max_volume, scenario_index)
            
            added_storage = max(0.0, v_new - v_cur)
            self.total_cost[scenario_index.global_id] = self.a * (added_storage ** self.b)

    @classmethod
    def load(cls, model, data):
        new_max_volume = data.pop("new_max_volume", data.pop("parameter", None))
        current_max_volume = data.pop("current_max_volume")
        a = data.pop("a")
        b = data.pop("b")
        return cls(model, new_max_volume, current_max_volume, a, b, **data)

ReservoirStorageExpansionCostRecorder.register()


class ApplicationEfficiencyCostRecorder(BaseCostRecorder):
    """Recorder to calculate the CAPEX of on-farm irrigation efficiency improvements.

    cost = C_max * ((e_a - e_0) / (e_max - e_0))^b * Area

    Attributes
    ----------
    efficiency_parameter : pywr.parameters.Parameter or float
        The proposed application efficiency fraction `e_a`.
    e0 : float
        The baseline application efficiency `e_0`.
    emax : float
        The maximum achievable application efficiency `e_max`.
    cmax : float
        The cost per hectare at maximum efficiency `C_max`.
    b : float
        The exponent function for increasing marginal costs.
    area_ha : pywr.parameters.Parameter or float
        The total on-farm area in hectares.
    """
    def __init__(self, model, efficiency_parameter, e0, emax, cmax, b, area_ha, **kwargs):
        super().__init__(model, **kwargs)
        self.efficiency_parameter = _load_param_or_float(model, efficiency_parameter)
        self.e0 = float(e0)
        self.emax = float(emax)
        self.cmax = float(cmax)
        self.b = float(b)
        self.area_ha = _load_param_or_float(model, area_ha)

        if isinstance(self.efficiency_parameter, Parameter):
            self.efficiency_parameter.parents.add(self)
        if isinstance(self.area_ha, Parameter):
            self.area_ha.parents.add(self)

    def after(self):
        return 0

    def finish(self):
        for scenario_index in self.model.scenarios.combinations:
            e_a = _get_val(self.efficiency_parameter, scenario_index)
            area = _get_val(self.area_ha, scenario_index)

            if e_a <= self.e0:
                cost_per_ha = 0.0
            else:
                ratio = (e_a - self.e0) / (self.emax - self.e0)
                # Clamp ratio to 1.0 safely in case e_a > emax
                ratio = min(1.0, ratio)
                cost_per_ha = self.cmax * (ratio ** self.b)
            
            self.total_cost[scenario_index.global_id] = cost_per_ha * area

    @classmethod
    def load(cls, model, data):
        efficiency_parameter = data.pop("efficiency_parameter", data.pop("parameter", None))
        e0 = data.pop("e0")
        emax = data.pop("emax")
        cmax = data.pop("cmax")
        b = data.pop("b")
        area_ha = data.pop("area_ha")
        return cls(model, efficiency_parameter, e0, emax, cmax, b, area_ha, **data)

ApplicationEfficiencyCostRecorder.register()


class IrrigationSchemeRehabilitationCostRecorder(BaseCostRecorder):
    """Recorder for assessing scheme-scale rehabilitation and modernization CAPEX.

    cost = m * (fixed + u * A^delta)

    Attributes
    ----------
    area_kha : pywr.parameters.Parameter or float
        The rehabilitated scheme command area `A` in thousand hectares (kha).
    fixed : float
        The fixed mobilization tracking cost envelope `F` (million USD).
    u : float
        The scheme scaling coefficient `u`.
    delta : float
        The scheme scaling exponent `delta`.
    m_multiplier : float
        The optional intensity multiplier `m` (defaults to 1.0).
    """

    def __init__(self, model, area_kha, fixed, u, delta, m_multiplier=1.0, **kwargs):
        super().__init__(model, **kwargs)
        self.area_kha = _load_param_or_float(model, area_kha)
        self.fixed = float(fixed)
        self.u = float(u)
        self.delta = float(delta)
        self.m_multiplier = float(m_multiplier)

        if isinstance(self.area_kha, Parameter):
            self.area_kha.parents.add(self)

    def after(self):
        return 0

    def finish(self):
        for scenario_index in self.model.scenarios.combinations:
            area = _get_val(self.area_kha, scenario_index)
            
            if area <= 0.0:
                self.total_cost[scenario_index.global_id] = 0.0
            else:
                self.total_cost[scenario_index.global_id] = self.m_multiplier * (self.fixed + self.u * (area ** self.delta))

    @classmethod
    def load(cls, model, data):
        area_kha = data.pop("area_kha", data.pop("parameter", None))
        fixed = data.pop("fixed")
        u = data.pop("u")
        delta = data.pop("delta")
        m_multiplier = data.pop("m_multiplier", 1.0)
        return cls(model, area_kha, fixed, u, delta, m_multiplier=m_multiplier, **data)

IrrigationSchemeRehabilitationCostRecorder.register()


# =====================================================================================
# This is the original implementation of the cost functions
# =====================================================================================
# class ConveyanceEfficiencyCostRecorder(BaseConstantParameterRecorder):
#     """Recorder to calculate the cost of improving canal conveyance efficiency.

#     This recorder calculates the total capital cost of improving canal conveyance
#     efficiency for a given length of canal. The cost is calculated using a
#     power-law relationship between the efficiency gain and the investment cost.

#     Attributes
#     ----------
#     parameter : pywr.parameters.Parameter
#         A parameter that provides the target conveyance efficiency as a fraction (0-1).
#     a : float
#         The cost coefficient.
#     b : float
#         The exponent of the cost function.
#     n0 : float
#         The baseline conveyance efficiency (fraction).
#     canal_length_km : float
#         The total length of the canal in kilometers.
#     """

#     def __init__(self, model, parameter, a, b, n0, canal_length_km, **kwargs):

#         agg_func = kwargs.pop('agg_func', 'mean')

#         super().__init__(model, parameter, **kwargs)
#         self.a = a
#         self.b = b
#         self.n0 = n0
#         self.canal_length_km = canal_length_km

#         self._scenario_aggregator = Aggregator(agg_func)
#         self._scenario_aggregator.func = agg_func

#     def setup(self):
#         ncomb = len(self.model.scenarios.combinations)
#         self.total_cost = np.zeros((ncomb))

#     def reset(self):
#         self.total_cost[...] = 0.0
        
#     def after(self):

#         for scenario_index in self.model.scenarios.combinations:
#             efficiency = 1 - self._param.get_value(scenario_index)
#             delta_efficiency = np.maximum(0.0, efficiency - (1-self.n0))
#             cost_per_km = self.a * (delta_efficiency ** self.b)
            
#             self.total_cost[scenario_index.global_id] = cost_per_km * self.canal_length_km

#         return 0

#     def values(self):
#         """Compute a value for each scenario using `temporal_agg_func`."""
#         return self.total_cost
    
#     def aggregated_value(self):
#         return self._scenario_aggregator.aggregate_1d(self.total_cost, ignore_nan=self.ignore_nan)


# ConveyanceEfficiencyCostRecorder.register()


# class ReservoirStorageExpansionCostRecorder(BaseConstantParameterRecorder):
#     """Recorder to calculate the cost of new reservoir storage.

#     This recorder calculates the capital cost of building new reservoir storage
#     capacity. The cost is calculated using a power function that captures
#     economies of scale.

#     Attributes
#     ----------
#     parameter : pywr.parameters.Parameter
#         A parameter that provides the additional storage capacity in MCM.
#     a : float
#         The coefficient of the cost function.
#     b : float
#         The exponent of the cost function.
#     """

#     def __init__(self, model, parameter, current_max_volumne, a, b, **kwargs):

#         agg_func = kwargs.pop('agg_func', 'mean')
#         super().__init__(model, parameter, **kwargs)
#         self.a = a
#         self.b = b

#         self.current_max_volumne = current_max_volumne

#         self._scenario_aggregator = Aggregator(agg_func)
#         self._scenario_aggregator.func = agg_func

#     def setup(self):
#         ncomb = len(self.model.scenarios.combinations)
#         self.total_cost = np.zeros((ncomb))

#     def reset(self):
#         self.total_cost[...] = 0.0

#     def after(self):

#         for scenario_index in self.model.scenarios.combinations:
#             storage = self._param.get_value(scenario_index) - self.current_max_volumne.get_value(scenario_index)
#             self.total_cost[scenario_index.global_id] = self.a * (storage ** self.b)

#         return 0

#     def values(self):
#         """Compute a value for each scenario using `temporal_agg_func`."""
#         return self.total_cost
    
#     def aggregated_value(self):
#         return self._scenario_aggregator.aggregate_1d(self.total_cost, ignore_nan=self.ignore_nan)
    

#     @classmethod
#     def load(cls, model, data):
#         """Loads the recorder from a dictionary configuration."""

#         parameter = load_parameter(model, data.pop("parameter"))
#         current_max_volumne = load_parameter(model, data.pop("current_max_volumne"))
        
#         # Remaining items in data are passed as kwargs to __init__
#         return cls(model, parameter, current_max_volumne, **data)

# ReservoirStorageExpansionCostRecorder.register()


# =====================================================================================
# =====================================================================================
# Delete everything from here! those are Mikiyas parameters and recorders.
# =====================================================================================
# =====================================================================================

from pywr.recorders import *
from pywr.nodes import Storage, Input, Node, Output
from pywr.parameters import (Parameter, load_parameter, InterpolatedVolumeParameter, MonthlyProfileParameter, ConstantParameter,
                             AggregatedParameter)
from pywr.parameters._hydropower import inverse_hydropower_calculation
import scipy.interpolate

class Rabi_water_allocation_origional(Parameter):
    def __init__(self, model, Punjab_channel_heads, Sindh_channel_heads, input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, System_losses_percent_Indus, System_losses_JC, **kwargs):
        super().__init__(model, **kwargs)
        self.input_data = input_data
        self.Mangla_reservoir=Mangla_reservoir
        self.Punjab_channel_heads = Punjab_channel_heads
        self.Sindh_channel_heads = Sindh_channel_heads
        
        self.Punjab_channel_heads_node = {node_name:model._get_node_from_ref(model, node_name) for node_name in self.Punjab_channel_heads}
        self.Sindh_channel_heads_node = {node_name:model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads}
        self.Punjab_channel_heads_recorders_name = [node+"_rec" for node in self.Punjab_channel_heads]
        self.Sindh_channel_heads_recorders_name = [node+"_rec" for node in self.Sindh_channel_heads]
        self.Punjab_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Punjab_channel_heads_recorders_name}
        self.Sindh_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Sindh_channel_heads_recorders_name}
        
        self.Tarbela_reservoir=Tarbela_reservoir
        self.Indus_at_Chashma=Indus_at_Chashma
        self.Storage_Dep_at_end_of_season_Mangla=Storage_Dep_at_end_of_season_Mangla
        self.Storage_Dep_at_end_of_Season_Tarbela=Storage_Dep_at_end_of_Season_Tarbela
        self.percentage_range=percentage_range
        self.Filling_withdraw_fraction_Tarbela=Filling_withdraw_fraction_Tarbela
        self.Filling_withdraw_fraction_Mangla=Filling_withdraw_fraction_Mangla
        self.Eastern_rivers=Eastern_rivers
        self.JC_average_system_uses_1977_1982=JC_average_system_uses_1977_1982
        self.Average_System_use_Indus=Average_System_use_Indus
        self.KPK_Baloch_share=KPK_Baloch_share
        self.KPK_share_historical=KPK_share_historical
        self.Baloch_share_historical=Baloch_share_historical
        self.Below_Kotri=Below_Kotri
        self.Punjab_share_Indus_para_2_percent=Punjab_share_Indus_para_2_percent
        self.System_losses_percent_Indus=System_losses_percent_Indus
        self.System_losses_JC=System_losses_JC

    def setup(self):
        super().setup()
        
    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        ts = self.model.timestepper.current
        days_in_month = timestep.period.days_in_month
        start_date = str(ts.year)+"-"+str(ts.month)+"-"+str(ts.day)
        self.val = 0
        start_year = self.model.timestepper.start.year
        Initial_Storage_Mangla = self.Mangla_reservoir.volume[i]/43560
        Maximum_Storage_Mangla = self.Mangla_reservoir.max_volume/43560
        Initial_Storage_Tarbela = self.Tarbela_reservoir.volume[i]/43560
        Maximum_Storage_Tarbela = self.Tarbela_reservoir.max_volume/43560
        
        self.pubjab_abstructed_Rabi = {}
        self.sindh_abstructed_Rabi = {}
        
        self.pubjab_remaining_demand_Rabi = {}
        self.sindh_remaining_demand_Rabi = {}
        
        if ts.month == 9 and ts.day > 29:
            self.Rabi_season_start_date_index = ts.index
            
        if start_year < ts.year:
            if ts.month == 9 and ts.day > 29:
                self.Rabi_index = 0
                self.Rabi_season_start_date_index = ts.index
                water_balance_J_C_zone_output = water_balance_J_C_zone(self.input_data, Initial_Storage_Mangla, Maximum_Storage_Mangla, Initial_Storage_Tarbela, Maximum_Storage_Tarbela, self.Indus_at_Chashma, self.Storage_Dep_at_end_of_season_Mangla, self.Storage_Dep_at_end_of_Season_Tarbela, start_date, self.percentage_range, self.Filling_withdraw_fraction_Tarbela, self.Filling_withdraw_fraction_Mangla, self.Eastern_rivers, self.JC_average_system_uses_1977_1982, self.Average_System_use_Indus, self.KPK_Baloch_share, self.KPK_share_historical, self.Baloch_share_historical, self.Below_Kotri, self.Punjab_share_Indus_para_2_percent, self.System_losses_percent_Indus, self.System_losses_JC)
                
                self.Sindh_Channel_dis_df_likely = water_balance_J_C_zone_output["Sindh_Channel_dis_df_likely"] 
                self.Punjab_J_C_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_J_C_Channel_dis_df_likely"]
                self.Punjab_Indus_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_Indus_Channel_dis_df_likely"]
                self.RQBS_Canal_Outflow_likely = water_balance_J_C_zone_output["RQBS_Canal_Outflow_likely"]

                self.Sindh_total_allocated_water = self.Sindh_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_J_C_total_allocated_water = self.Punjab_J_C_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_Indus_total_allocated_water = self.Punjab_Indus_Channel_dis_df_likely.sum(axis=1).tolist()

            Rabi_start_time = pd.to_datetime(str(ts.year) + '-09-29')
            Rabi_end_time = pd.to_datetime(str(ts.year + 1) + '-03-30') 
            if Rabi_start_time <= ts.datetime <= Rabi_end_time:
                #water used so far
                self.pubjab_abstructed_Rabi = {recorder:sum(self.Punjab_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Punjab_channel_heads_recorders}
                self.sindh_abstructed_Rabi = {recorder:sum(self.Sindh_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Sindh_channel_heads_recorders}
                
                #remining water that is required to satisfy the full demand
                self.pubjab_remaining_demand_Rabi = {node:sum(self.Punjab_channel_heads_node[node].max_flow.dataframe[ts.datetime : Rabi_end_time].values) for node in self.Punjab_channel_heads_node}
                self.sindh_remaining_demand_Rabi  = {node:sum(self.Sindh_channel_heads_node[node].max_flow.dataframe[ts.datetime : Rabi_end_time].values) for node in self.Sindh_channel_heads_node}
                
                #how much avaiable is avaiable to allocate according to IRSA's prediction 
                
                if ts.day == 10 or ts.day == 20:
                    
                    self.Sindh_Channel_available_water = sum(self.Sindh_total_allocated_water[self.Rabi_index:])*43560
                    self.Punjab_J_C_Channel_available_water = sum(self.Punjab_J_C_total_allocated_water[self.Rabi_index:])
                    self.Punjab_Indus_Channel_available_water = sum(self.Punjab_Indus_total_allocated_water[self.Rabi_index:])
                    self.Rabi_index += 1
                    
                elif ts.day == days_in_month:
                    
                    self.Sindh_Channel_available_water = sum(self.Sindh_total_allocated_water[self.Rabi_index:])*43560
                    self.Punjab_J_C_Channel_available_water = sum(self.Punjab_J_C_total_allocated_water[self.Rabi_index:])
                    self.Punjab_Indus_Channel_available_water = sum(self.Punjab_Indus_total_allocated_water[self.Rabi_index:])
                    self.Rabi_index += 1
        
        #need the list of nodes in Punjab J-C and Indus Zone and Sindh Zone
        #check the next 183 days demand and if there less water reduce the demand by a certain fraction for those who are using more than the allocated amount
        #J-C outflow need to be tracked everytime and the value need to match with the allocated one
        
        #step1: collect all the recorders to get how much water is allocated 
        #step2: then create a number of 
             
        val = sum(self.pubjab_remaining_demand_Rabi.values())
        return val
            
    @classmethod
    def load(cls, model, data):
        Mangla_reservoir = model._get_node_from_ref(model, data.pop("Mangla_reservoir_node"))
        Tarbela_reservoir = model._get_node_from_ref(model, data.pop("Tarbela_reservoir_node"))
        
        Indus_at_Chashma = data.pop("Indus_at_Chashma")
        Storage_Dep_at_end_of_season_Mangla = data.pop("Storage_Dep_at_end_of_season_Mangla")
        Storage_Dep_at_end_of_Season_Tarbela = data.pop("Storage_Dep_at_end_of_Season_Tarbela")
        System_losses = data.pop("System_losses")
        
        percentage_range = data.pop("percentage_range")
        Filling_withdraw_fraction_Tarbela = data.pop("Filling_withdraw_fraction_Tarbela")
        Filling_withdraw_fraction_Mangla = data.pop("Filling_withdraw_fraction_Mangla")
        Eastern_rivers = data.pop("Eastern_rivers")
        JC_average_system_uses_1977_1982 = data.pop("JC_average_system_uses_1977_1982")
        Average_System_use_Indus = data.pop("Average_System_use_Indus")
        KPK_Baloch_share = data.pop("KPK_Baloch_share")
        KPK_share_historical = data.pop("KPK_share_historical")
        Baloch_share_historical = data.pop("Baloch_share_historical")
        Below_Kotri = data.pop("Below_Kotri")
        Punjab_share_Indus_para_2_percent = data.pop("Punjab_share_Indus_para_2_percent")
        System_losses_percent_Indus = data.pop("System_losses_percent_Indus")
        System_losses_JC = data.pop("System_losses_JC")
        input_data = data.pop("url")
        Punjab_channel_heads = data.pop("Punjab_channel_heads")
        Sindh_channel_heads = data.pop("Sindh_channel_heads")
        
        return cls(model, Punjab_channel_heads, Sindh_channel_heads, input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, System_losses_percent_Indus, System_losses_JC, **data)  

Rabi_water_allocation_origional.register()


# class RootMeanSquaredErrorNodeRecorder_(NumpyArrayNodeRecorder):

#     def __init__(self, model, node, observed,index, **kwargs):
#         super(RootMeanSquaredErrorNodeRecorder_, self).__init__(model, node, **kwargs)
#         self.observed = pd.read_hdf(observed)[index]
#         self._aligned_observed = None
        

#     def setup(self):
#         super(RootMeanSquaredErrorNodeRecorder_, self).setup()
#         # Align the observed data to the model
#         self._aligned_observed = align_and_resample_dataframe(self.observed, self.model.timestepper.datetime_index)

#     def values(self):
#         mod = self.data
#         obs = self._aligned_observed
#         return np.sqrt(np.mean((obs-mod)**2, axis=0))

#     @classmethod
#     def load(cls, model, data):
#         observed = data.pop("observed")   
#         index = data.pop("index")      
#         node = model._get_node_from_ref(model, data.pop("node"))    
        
#         return cls(model, node, observed,index,**data)
    
# RootMeanSquaredErrorNodeRecorder_.register()


# class NashSutcliffeEfficiencyNodeRecorder_(NodeRecorder):

#     def __init__(self, model, node, observed, index, record_year, **kwargs):
#         super(NashSutcliffeEfficiencyNodeRecorder_, self).__init__(model, node, **kwargs)
#         self._aligned_observed = None
#         temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
#         factor = kwargs.pop('factor', 1.0)
#         self.factor = factor   
#         self._node = node
#         self.start=pd.to_datetime(str(record_year)+'-01-01 00:00:00')
#         self.end=pd.to_datetime(str(2009)+'-12-31 00:00:00')
#         self.observed = pd.read_hdf(observed)[index][ self.start:self.end]


#     def setup(self):
#         ncomb = len(self.model.scenarios.combinations)
#         nts = len(self.model.timestepper)
#         self._data = np.zeros((nts))
#         self.nsc = np.zeros((nts))
#         # Align the observed data to the model
#         self._aligned_observed = self.observed

#     def reset(self):
#         self._data[:] = 0.0

#     def after(self):
#         ts = self.model.timestepper.current
#         #for i in range(self._data.shape[0]):
#         self._data[ts.index] = self._node.flow[0]
#         mod = self._data[-len(self._aligned_observed.values):]
#         obs = self._aligned_observed
#         obs_mean = np.mean(obs, axis=0)
#         self.nse_ = 1.0 - np.sum((obs-mod)**2, axis=0)/np.sum((obs-obs_mean)**2, axis=0)

#         self.nsc[ts.index] = self.nse_
    
        
#     def values(self):
#         """Compute a value for each scenario using `temporal_agg_func`."""
#         self.NSE_obj=np.zeros((1))
#         self.NSE_obj[0]=self.nse_
#         return self.NSE_obj

#     def to_dataframe(self):
#         """ Return a `pandas.DataFrame` of the recorder data
#         This DataFrame contains a MultiIndex for the columns with the recorder name
#         as the first level and scenario combination names as the second level. This
#         allows for easy combination with multiple recorder's DataFrames
#         """
#         index = self.model.timestepper.datetime_index
#         sc_index = self.model.scenarios.multiindex

#         return pd.DataFrame(data=np.array(self.nsc), index=index, columns=sc_index)

#     @classmethod
#     def load(cls, model, data):
#         observed = data.pop("observed")   
#         index = data.pop("index")      
#         node = model._get_node_from_ref(model, data.pop("node"))    
#         record_year = data.pop("record_year")
#         return cls(model, node, observed, index, record_year, **data)
    
# NashSutcliffeEfficiencyNodeRecorder_.register()


# class MeanSquareErrorNodeRecorder(NodeRecorder):

#     def __init__(self, model, node, observed, index, record_year, **kwargs):
#         super(MeanSquareErrorNodeRecorder, self).__init__(model, node, **kwargs)
#         self._aligned_observed = None
#         temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
#         factor = kwargs.pop('factor', 1.0)
#         self.factor = factor   
#         self._node = node
#         self.start=pd.to_datetime(str(record_year)+'-01-01 00:00:00')
#         self.end=pd.to_datetime(str(2009)+'-12-31 00:00:00')
#         self.observed = pd.read_hdf(observed)[index][ self.start:self.end]


#     def setup(self):
#         ncomb = len(self.model.scenarios.combinations)
#         nts = len(self.model.timestepper)
#         self._data = np.zeros((nts))
#         self.nsc = np.zeros((nts))
#         # Align the observed data to the model
#         self._aligned_observed = self.observed


#     def reset(self):
#         self._data[:] = 0.0

#     def after(self):
#         ts = self.model.timestepper.current
#         #for i in range(self._data.shape[0]):
#         self._data[ts.index] = self._node.flow[0]
#         mod = self._data[-len(self._aligned_observed.values):]
#         obs = self._aligned_observed
#         self.RMSE = np.mean((obs-mod)**2, axis=0)
        

#         return self.RMSE
        
#     def values(self):
#         """Compute a value for each scenario using `temporal_agg_func`."""
#         self.NSE_obj=np.zeros((1))
#         self.NSE_obj[0]=self.RMSE

#         return self.NSE_obj


#     def to_dataframe(self):
#         """ Return a `pandas.DataFrame` of the recorder data
#         This DataFrame contains a MultiIndex for the columns with the recorder name
#         as the first level and scenario combination names as the second level. This
#         allows for easy combination with multiple recorder's DataFrames
#         """
#         index = self.model.timestepper.datetime_index
#         sc_index = self.model.scenarios.multiindex

#         return pd.DataFrame(data=np.array(self.nsc), index=index, columns=sc_index)

#     @classmethod
#     def load(cls, model, data):
#         observed = data.pop("observed")   
#         index = data.pop("index")      
#         node = model._get_node_from_ref(model, data.pop("node"))    
#         record_year = data.pop("record_year")
#         return cls(model, node, observed, index, record_year, **data)
    
# MeanSquareErrorNodeRecorder.register()


class Simple_Irr_demand_calculator(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, evaporation,rainfall_factor,IRR_exp,IRR_eff,crop_ET,crop_area,Max_area,rainfall,index_col, **kwargs):
        super().__init__(model, **kwargs)
        self.evaporation = [x/30 for x in evaporation]
        self.rainfall_factor = rainfall_factor
        self.IRR_exp = IRR_exp
        self.IRR_eff = IRR_eff
        self.crop_ET = crop_ET
        self.crop_area = crop_area
        self.Max_area = Max_area
        self.rainfall_ = rainfall
        self.index_col=index_col
        
    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        self.rainfall_=pd.read_hdf(self.rainfall_)
        
        
    def value(self, timestep, scenario_index):    
        
        ts = self.model.timestepper.current
        month=ts.month
        year=ts.year
        self.rainfall=self.rainfall_[self.index_col][str(year)+"_"+str(month)]/ts.day
        
        rainfall_factor=self.rainfall_factor
                
        evaporation=self.evaporation[month-1]
        Total_Irr_demand=0
        for x in self.crop_ET.keys():
            if x in self.crop_area.keys():
                Net_demand=max((self.crop_ET[x][month-1]*evaporation-rainfall_factor[month-1]*self.rainfall),0)
                Irr_eff=1+(1-0.8)
                Irr_demand=Net_demand*(self.crop_area[x]*0.01)*self.Max_area*Irr_eff * 1e6 * 1e-3 * 1e-6
                
                Total_Irr_demand+=Irr_demand
        
        return Total_Irr_demand
        

    @classmethod
    def load(cls, model, data):
        evaporation = data.pop("evaporation")
        rainfall_factor = data.pop("rainfall_factor")
        IRR_exp = data.pop("IRR_exp")
        IRR_eff = data.pop("IRR_eff")
        crop_ET = data.pop("crop_ET")
        crop_area = data.pop("crop_area")
        Max_area = data.pop("Max_area")
        rainfall = data.pop("rainfall")
        index_col=data.pop("index_col")
        
        
        return cls(model, evaporation,rainfall_factor,IRR_exp,IRR_eff,crop_ET,crop_area,Max_area,rainfall, index_col,**data)
    
Simple_Irr_demand_calculator.register()


class Reservoir(Storage):
    def __init__(self, model, name, **kwargs):
        #where are you guys, the input and output nodes connected to self?
        #level = kwargs.pop('levels', None)
        volume = kwargs.pop('volumes', None)
        area = kwargs.pop('areas', None)
        rainfall= kwargs.pop('rainfall', None)
        evaporation= kwargs.pop('evaporation', None)
        weather_cost = kwargs.pop('weather_cost', -99999999)
        super().__init__(model, name, **kwargs)
        self._set_bathymetry(area, volume)
        self.rainfall_node = None
        self.evaporation_node = None
        
        if rainfall is not None:
            self._make_weather_nodes(model, rainfall, evaporation, weather_cost)
            
            
    def _set_bathymetry(self, areas,volumes):
        #self.level = InterpolatedVolumeParameter(self.model, self, volumes, levels)
        self.area = InterpolatedVolumeParameter(self.model, self, volumes, areas)
 

            
    def _make_weather_nodes(self, model, rainfall, evaporation,cost):
        if not isinstance(self.area, Parameter):
            raise ValueError('Weather nodes can only be created if an area Parameter is given.')

        rainfall_param = MonthlyProfileParameter(model, rainfall)
        evaporation_param = MonthlyProfileParameter(model, evaporation)

        # Assume rainfall/evap is mm/day
        # Need to convert:
        #   Mm2 -> m2
        #   mm/day -> m/day
        #   m3/day -> Mm3/day
        # TODO allow this to be configured
        const = ConstantParameter(model, 1e6 * 1e-3 * 1e-6)

        # Create the flow parameters multiplying area by rate of rainfall/evap
        rainfall_flow_param = AggregatedParameter(model, [rainfall_param, const, self.area],
                                                  agg_func='product')
        evaporation_flow_param = AggregatedParameter(model, [evaporation_param, const, self.area],
                                                     agg_func='product')
        
        # Create the nodes to provide the flows
        rainfall_node = Input(model, '{}.rainfall'.format(self.name), parent=self)
        rainfall_node.max_flow = rainfall_flow_param
        
        rainfall_node.cost = cost

        evporation_node = Output(model, '{}.evaporation'.format(self.name), parent=self)
        evporation_node.max_flow = evaporation_flow_param

        evporation_node.cost = cost
        
        rainfall_node.connect(self)
        self.connect(evporation_node)
        self.rainfall_node = rainfall_node
        self.evaporation_node = evporation_node

        # Finally record these flows
        #self.rainfall_recorder = NumpyArrayNodeRecorder(model, rainfall_node, name=f'__{rainfall_node.name}__:rainfall')
        #self.evaporation_recorder = NumpyArrayNodeRecorder(model, evporation_node, name=f'__{evporation_node.name}__:evaporation')    


class res_area_param(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, Area_factor, Reservoir_, **kwargs):
        super().__init__(model, **kwargs)
        self.Area_factor = Area_factor
        self.Reservoir_ = Reservoir_
        
        
    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
    def value(self, timestep, scenario_index):    
        i = scenario_index.global_id
        
        Res_Volume = self.Reservoir_.volume[i]
        
        ts = self.model.timestepper.current        
        Res_Area=(Res_Volume**self.Area_factor)*1e-3
    
        return Res_Area
        

    @classmethod
    def load(cls, model, data):
        Area_factor = data.pop("Area_factor")
        Reservoir_ = data.pop("Reservoir")
        Reservoir_=model._get_node_from_ref(model, Reservoir_)
        return cls(model, Area_factor, Reservoir_, **data)
res_area_param.register()

class res_rainfall_param(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, url, index_col, column, Area_factor, Reservoir, **kwargs):
        super().__init__(model, **kwargs)
        self.url = pd.read_hdf(url)
        self.index_col = index_col
        self.column = column
        self.Area_factor = Area_factor
        self.Reservoir = Reservoir
        
        
    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
    def value(self, timestep, scenario_index):    
        i = scenario_index.global_id
        Res_Volume = self.Reservoir.volume[i]
        
        ts = self.model.timestepper.current
        
        month=ts.month
        year=ts.year
        
        rainfall=self.url[self.column][str(year)+"_"+str(month)]/ts.day
        
        Rain_volume=(Res_Volume**self.Area_factor)*rainfall*1e-3
        
    
        return Rain_volume
        

    @classmethod
    def load(cls, model, data):
        url = data.pop("url")
        index_col = data.pop("index_col")
        column = data.pop("column")
        Area_factor = data.pop("Area_factor")
        Reservoir = data.pop("Reservoir")
        Reservoir = model._get_node_from_ref(model, Reservoir)
        
        return cls(model, url, index_col, column, Area_factor, Reservoir, **data)
res_rainfall_param.register()

class res_vol_param(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, volume, **kwargs):
        super().__init__(model, **kwargs)
        self.volume = volume

        
    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        
        
    def value(self, timestep, scenario_index):        
        ts = self.model.timestepper.current
        year=1990
        years_=[int(x) for x in self.volume.keys()]
        volume_cap_year=sorted(i for i in years_ if i <= year)[0]
        volume=self.volume[str(volume_cap_year)]["volume"]

        return volume
        
    @classmethod
    def load(cls, model, data):
        volume = data.pop("volume")
        return cls(model, volume, **data)
res_vol_param.register()


class Rim_station_flow_forcast(Parameter):
    def __init__(self, model, historica_data, node_, probablity_table, rim_station, **kwargs):
        super().__init__(model, **kwargs)
        self.Flow_rim_station = pd.read_csv(historica_data)
        self.probablity_table = pd.read_csv(probablity_table)
        self.node_ = node_
        self.rim_station = rim_station
        self.percentage_range = 0.05

    def setup(self):
        super().setup()
        #self.node_ = self.model.recorders[self.node_]
        del self.probablity_table['Ten days']
        del self.probablity_table['Month']
        self.probablity_table = self.probablity_table.astype(float)
        self.Flow_rim_station.index = self.Flow_rim_station['Year']
        del self.Flow_rim_station['Year']
        self.prediction = np.zeros(18)
        self.end_of_Rabi = self.probablity_table.loc[17]
        self.end_of_Kharif =  self.probablity_table.loc[35]

    def calculate_matching_years(self, node_prev_flow, rim_station):
        #self.prediction should come from the node result for the past 6 month or the t-1 timestep (need to check)
        lower_bound = node_prev_flow - (node_prev_flow * self.percentage_range)
        upper_bound = node_prev_flow + (node_prev_flow * self.percentage_range)

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@", lower_bound, upper_bound)
        #filter matching years
        filtered_df = self.Flow_rim_station[(self.Flow_rim_station[rim_station] >= lower_bound) & (self.Flow_rim_station[rim_station] <= upper_bound)]
        #average flows of matching years
        self.mean_value = filtered_df[rim_station].mean()
        

        return self.mean_value

    def probabilities_average_flow(self, end_of_season_flow, mean_value, season):
        # Calculate the absolute differences from the given value x

        if season == "Kharif":
            end_of_Rabi = self.probablity_table.loc[17]
            differences = abs(end_of_Rabi - mean_value)

            # Find the column with the minimum difference
            min_diff_column = differences.idxmin()
            
            prediction = self.probablity_table[min_diff_column][18:36]

        else:
            end_of_Kharif = self.probablity_table.loc[35]
            differences = abs(end_of_Kharif - mean_value)
            # Find the column with the minimum difference
            min_diff_column = differences.idxmin()
            prediction = self.probablity_table[min_diff_column][0:18]
        #should return the predected flow/storage_volume which should be a value rather than array
     
        return prediction

    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current
        #there are two seasons Rabi (from April 1 to Sep 20) and Kharif (Oct to March)

        if ts.year > 1991:
            if ts.month == 9 and ts.day > 28:
                season = "Rabi"
                node_prev_flow = np.sum(self.node_.data[ts.index - 180:ts.index])
                
                rim_station = self.rim_station + "_Rabi"
                mean_value = self.calculate_matching_years(node_prev_flow, rim_station)
                #mean_value = 14.847
                self.prediction = list(self.probabilities_average_flow(node_prev_flow, mean_value, season))

            elif ts.month == 3 and ts.day > 28: 
                
                season = "Kharif"
                node_prev_flow = np.sum(self.node_.data[ts.index - 180:ts.index])
                rim_station = self.rim_station + "_Kharif" 
                mean_value = self.calculate_matching_years(node_prev_flow, rim_station)
                #mean_value = 14.847
                self.prediction = list(self.probabilities_average_flow(node_prev_flow, mean_value, season))
            return self.prediction[0]
        else:
            return 0

    @classmethod
    def load(cls, model, data):
        historica_data = data.pop("historica_data")   
        node_ = load_recorder(model, data.pop("node"))      
        probablity_table = data.pop("probablity_table")
        rim_station = data.pop("rim_station")
        return cls(model, historica_data, node_, probablity_table, rim_station, **data)
Rim_station_flow_forcast.register()


class Reservoir_operation_5_years_average(Parameter):
    def __init__(self, model, historica_release, reservoir_name, **kwargs):
        super().__init__(model, **kwargs)
        self.historica_release = pd.read_hdf(historica_release)
        self.reservoir_name = reservoir_name

    def setup(self):
        super().setup()
        daily_mean_1994 = self.historica_release['1990-01-01':'1994-12-31'][self.reservoir_name]
        daily_mean_1998 = self.historica_release['1995-01-01':'1998-12-31'][self.reservoir_name]
        daily_mean_2003 = self.historica_release['1999-01-01':'2003-12-31'][self.reservoir_name]
        daily_mean_2008 = self.historica_release['2004-01-01':'2008-12-31'][self.reservoir_name]
        daily_mean_2013 = self.historica_release['2009-01-01':'2013-12-31'][self.reservoir_name]
        daily_mean_2018 = self.historica_release['2014-01-01':'2018-12-31'][self.reservoir_name]

        self.daily_mean_1994 = daily_mean_1994.groupby(daily_mean_1994.index.dayofyear).mean()
        self.daily_mean_1998 = daily_mean_1998.groupby(daily_mean_1998.index.dayofyear).mean()
        self.daily_mean_2003 = daily_mean_2003.groupby(daily_mean_2003.index.dayofyear).mean()
        self.daily_mean_2008 = daily_mean_2008.groupby(daily_mean_2008.index.dayofyear).mean()
        self.daily_mean_2013 = daily_mean_2013.groupby(daily_mean_2013.index.dayofyear).mean()
        self.daily_mean_2018 = daily_mean_2018.groupby(daily_mean_2018.index.dayofyear).mean()


    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current
        day = ts.dayofyear
        year = ts.year
        #there are two seasons Rabi (from April 1 to Sep 20) and Kharif (Oct to March)

        if year >= 1990 and year <= 1994:
            release = self.daily_mean_1994[day]
        elif year > 1994 and year <= 1998:
            release = self.daily_mean_1998[day]
        elif year > 1998 and year <= 2003:
            release = self.daily_mean_2003[day]
        elif year > 2003 and year <= 2008:
            release = self.daily_mean_2008[day]
        elif year > 2008 and year <= 2013:
            release = self.daily_mean_2013[day]
        else:
            release = self.daily_mean_2018[day]
        return release

    @classmethod
    def load(cls, model, data):
        historica_release = data.pop("historica_release")   
        reservoir_name = data.pop("reservoir_name")    
        return cls(model, historica_release, reservoir_name, **data)
Reservoir_operation_5_years_average.register()

class Reservoir_operation_10_years_average(Parameter):
    def __init__(self, model, historica_release, reservoir_name, **kwargs):
        super().__init__(model, **kwargs)
        self.historica_release = pd.read_hdf(historica_release)
        self.reservoir_name = reservoir_name

    def setup(self):
        super().setup()
        daily_mean_1994 = self.historica_release['1990-01-01':'1998-12-31'][self.reservoir_name]
        daily_mean_2003 = self.historica_release['1999-01-01':'2008-12-31'][self.reservoir_name]
        daily_mean_2013 = self.historica_release['2009-01-01':'2018-12-31'][self.reservoir_name]

        self.daily_mean_1994 = daily_mean_1994.groupby(daily_mean_1994.index.dayofyear).mean()
        self.daily_mean_2003 = daily_mean_2003.groupby(daily_mean_2003.index.dayofyear).mean()
        self.daily_mean_2013 = daily_mean_2013.groupby(daily_mean_2013.index.dayofyear).mean()

    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current
        day = ts.dayofyear
        year = ts.year
        #there are two seasons Rabi (from April 1 to Sep 20) and Kharif (Oct to March)

        if year >= 1990 and year <= 1998:
            release = self.daily_mean_1994[day]
        elif year > 1998 and year <= 2008:
            release = self.daily_mean_2003[day]
        else:
            release = self.daily_mean_2013[day]
        return release 

    @classmethod
    def load(cls, model, data):
        historica_release = data.pop("historica_release")   
        reservoir_name = data.pop("reservoir_name")    
        return cls(model, historica_release, reservoir_name, **data)
Reservoir_operation_10_years_average.register()


class Reservior_operation_matching_year(Parameter):
    def __init__(self, model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, **kwargs):
        super().__init__(model, **kwargs)
        self.historical_flow_df = Historical_release_wl
        self.Storage_node = Storage_node
        self.inflow = inflow
        self.inflow_column = inflow_column
        self.outflow_column = outflow_column
        self.reservoir_level_column = reservoir_level_column


    def setup(self):
        super().setup()
        self.historical_flow_df = pd.read_hdf(self.historical_flow_df)
        self.Res_inflow = self.historical_flow_df[self.inflow_column]
        self.Res_outflow = self.historical_flow_df[self.outflow_column]
        self.Res_level = self.historical_flow_df[self.reservoir_level_column] 

        

    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current


        i = scenario_index.global_id 
        current_water_level = self.Storage_node.value(timestep, scenario_index)
        current_inflow = self.inflow.value(timestep, scenario_index)


        # Calculate distance
        self.historical_flow_df["distance"]  = np.sqrt((self.Res_level - current_water_level)**2 + (self.Res_inflow - current_inflow)**2)

        # Find the row with the minimum distance
        nearest_row = self.historical_flow_df .loc[self.historical_flow_df["distance"].idxmin()]

        # Extract release from the nearest row
        nearest_release = nearest_row[self.outflow_column]
        
        return nearest_release


    @classmethod
    def load(cls, model, data):

        Historical_release_wl = data.pop("Historical_release_wl")
        reservoir_level_column = data.pop("reservoir_level_column")    
        inflow_column = data.pop("inflow_column") 
        outflow_column = data.pop("outflow_column")    

        Storage_node = load_parameter(model, data.pop("Storage_node"))
        inflow = load_parameter(model, data.pop("inflow_node")) 

        return cls(model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, **data)
Reservior_operation_matching_year.register()


class Reservior_operation_matching_year_2(Parameter):
    def __init__(self, model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, **kwargs):
        super().__init__(model, **kwargs)
        self.historical_flow_df = Historical_release_wl
        self.Storage_node = Storage_node
        self.inflow = inflow
        self.inflow_column = inflow_column
        self.outflow_column = outflow_column
        self.reservoir_level_column = reservoir_level_column


    def setup(self):
        super().setup()
        self.historical_flow_df = pd.read_hdf(self.historical_flow_df)
        self.Res_inflow = self.historical_flow_df[self.inflow_column]
        self.Res_outflow = self.historical_flow_df[self.outflow_column]
        self.Res_level = self.historical_flow_df[self.reservoir_level_column] 

        

    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current


        i = scenario_index.global_id 
        current_water_level = self.Storage_node.value(timestep, scenario_index)

        current_inflow = self.Res_inflow[str(ts.datetime.date())] 
        
        


        # Calculate distance
        self.historical_flow_df["distance"]  = np.sqrt((self.Res_level - current_water_level)**2 + (self.Res_inflow - current_inflow)**2)

        # Find the row with the minimum distance
        nearest_row = self.historical_flow_df.loc[self.historical_flow_df["distance"].idxmin()]

        

        # Extract release from the nearest row
        nearest_release = nearest_row[self.outflow_column]
        
        return nearest_release


    @classmethod
    def load(cls, model, data):

        Historical_release_wl = data.pop("url")
        reservoir_level_column = data.pop("reservoir_level_column")    
        inflow_column = data.pop("inflow_column") 
        outflow_column = data.pop("outflow_column")    
        Storage_node = load_parameter(model, data.pop("Storage_node"))
        inflow = model._get_node_from_ref(model, data.pop("inflow_node")) 
        

        return cls(model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, **data)
Reservior_operation_matching_year_2.register()


###################################################################################################################
###################################################################################################################

class Reservior_operation_matching_year_average(Parameter):
    def __init__(self, model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, number_years, **kwargs):
        super().__init__(model, **kwargs)
        self.historical_flow_df = Historical_release_wl
        self.Storage_node = Storage_node
        self.inflow = inflow
        self.inflow_column = inflow_column
        self.outflow_column = outflow_column
        self.reservoir_level_column = reservoir_level_column
        self.number_years = number_years

    def setup(self):
        super().setup()
        self.historical_flow_df = pd.read_hdf(self.historical_flow_df)
        self.Res_inflow = self.historical_flow_df[self.inflow_column]
        self.Res_outflow = self.historical_flow_df[self.outflow_column]
        self.Res_level = self.historical_flow_df[self.reservoir_level_column]
        
        mask = ~((self.historical_flow_df.index >= '1997-10-01') & (self.historical_flow_df.index <= '2003-04-01'))
        
        self.Res_inflow_MG_NF = self.historical_flow_df[self.inflow_column][mask]
        self.Res_outflow_MG_NF = self.historical_flow_df[self.outflow_column][mask]
        self.Res_level_MG_NF = self.historical_flow_df[self.reservoir_level_column][mask]
        
        self.Res_inflow_MG_LF = self.historical_flow_df[self.inflow_column]['01/10/1997':'01/04/2003']
        self.Res_outflow_MG_LF = self.historical_flow_df[self.outflow_column]['01/10/1997':'01/04/2003']
        self.Res_level_MG_LF = self.historical_flow_df[self.reservoir_level_column]['01/10/1997':'01/04/2003']

    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current

        i = scenario_index.global_id 
        
        if (ts.year > 1997) and (ts.year < 2003):
            Res_level_n = self.Res_level_MG_LF[self.Res_level_MG_LF.index.month == ts.month]
            Res_inflow_n = self.Res_inflow_MG_LF[self.Res_inflow_MG_LF.index.month == ts.month]
            
            current_water_level = self.Storage_node.value(timestep, scenario_index)
            #current_inflow = self.inflow.flow
            current_inflow = self.Res_inflow[str(ts.datetime.date())] 
            
            # Calculate distance
            self.historical_flow_df["distance"]  = np.sqrt((Res_level_n - current_water_level)**2 + (Res_inflow_n - current_inflow)**2)
    
            historical_flow_df_sorted = self.historical_flow_df.sort_values(by=['distance'])
            nearest_release = historical_flow_df_sorted.iloc[:4][self.outflow_column].mean()
        
        else:   
        
            Res_level_n = self.Res_level_MG_NF[self.Res_level_MG_NF.index.month == ts.month]
            Res_inflow_n = self.Res_inflow_MG_NF[self.Res_inflow_MG_NF.index.month == ts.month]
            
            current_water_level = self.Storage_node.value(timestep, scenario_index)
            #current_inflow = self.inflow.flow
            current_inflow = self.Res_inflow[str(ts.datetime.date())] 
            
            # Calculate distance
            self.historical_flow_df["distance"]  = np.sqrt((Res_level_n - current_water_level)**2 + (Res_inflow_n - current_inflow)**2)
    
            historical_flow_df_sorted = self.historical_flow_df.sort_values(by=['distance'])
            nearest_release = historical_flow_df_sorted.iloc[:self.number_years][self.outflow_column].mean()
        
        return nearest_release


    @classmethod
    def load(cls, model, data):

        Historical_release_wl = data.pop("url")
        reservoir_level_column = data.pop("reservoir_level_column")    
        inflow_column = data.pop("inflow_column") 
        outflow_column = data.pop("outflow_column")    
        Storage_node = load_parameter(model, data.pop("Storage_node"))
        inflow = model._get_node_from_ref(model, data.pop("inflow_node")) 
        number_years =  data.pop("number_years")  
        
        return cls(model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, number_years, **data)

Reservior_operation_matching_year_average.register()


class River_seasonal_loss(Parameter):
    def __init__(self, model, loss_early_Rabi, loss_late_Rabi, loss_early_Kharif, loss_late_Khraif, loss_node, **kwargs):
        super().__init__(model, **kwargs)
        self.loss_node=loss_node 
        self.loss_early_Rabi = loss_early_Rabi
        self.loss_late_Rabi = loss_late_Rabi
        self.loss_early_Kharif = loss_early_Kharif
        self.loss_late_Khraif = loss_late_Khraif


    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        node_flow=self.loss_node.prev_flow[i]
        ts = self.model.timestepper.current
        
        
        
        # the loss values are in fraction not in percentage
        if ts.month >= 10 and ts.month <= 12:
            
            if self.loss_early_Rabi < 0:
                loss = 0
            else:
                loss = node_flow*(self.loss_early_Rabi)
        elif ts.month >= 1 and ts.month <= 3:
            if self.loss_late_Rabi < 0:
                loss = 0
            else:
                loss = node_flow*(self.loss_late_Rabi)
        elif ts.month >= 4 and ts.month <= 6:
            if self.loss_early_Kharif < 0:
                loss = 0
            else:
                loss = node_flow*(self.loss_early_Kharif)
        elif ts.month >= 7 and ts.month <= 9:
            if self.loss_late_Khraif < 0:
                loss = 0
            else:
                loss = node_flow*(self.loss_late_Khraif)
            
        
        return loss
            
    @classmethod
    def load(cls, model, data):
        loss_early_Rabi = data.pop("loss_early_Rabi")
        loss_late_Rabi = data.pop("loss_late_Rabi")
        loss_early_Kharif = data.pop("loss_early_Kharif")
        loss_late_Khraif = data.pop("loss_late_Khraif")
        loss_node=model._get_node_from_ref(model, data.pop("Loss_node"))
        
        return cls(model, loss_early_Rabi, loss_late_Rabi, loss_early_Kharif, loss_late_Khraif, loss_node, **data)
        
River_seasonal_loss.register()


class Rim_station_flow_forcast(Parameter):
    def __init__(self, model, Punjab_channel_heads_node, reservoir_test, Historical_flow_Rim_station, IRSA_probablity_table, rim_station, node_, punjab_demand_nodes, sindh_demand_nodes, **kwargs):
        super().__init__(model, **kwargs)
        self.Historical_flow_Rim_station = Historical_flow_Rim_station
        self.IRSA_probablity_table = IRSA_probablity_table
        self.node_ = node_
        self.reservoir_test = reservoir_test
        self.Punjab_channel_heads_node = Punjab_channel_heads_node
        self.rim_station = rim_station
        self.punjab_demand_nodes = punjab_demand_nodes 
        self.sindh_demand_nodes = sindh_demand_nodes
        self.percentage_range = 0.05

    def setup(self):
        super().setup()
        self.Rabi_index = 0
        self.Kharif_index = 0
        self.Rim_station_data = pd.HDFStore(self.Historical_flow_Rim_station)
        self.Flow_rim_station = self.Rim_station_data['/Historical_flow']
        self.probablity_table = pd.HDFStore(self.IRSA_probablity_table)
        self.prediction = np.zeros(18)

    def calculate_matching_years(self, node_prev_flow, rim_station):
        lower_bound = node_prev_flow - (node_prev_flow * self.percentage_range)
        upper_bound = node_prev_flow + (node_prev_flow * self.percentage_range)
        filtered_df = self.Flow_rim_station[(self.Flow_rim_station[rim_station] >= lower_bound) & (self.Flow_rim_station[rim_station] <= upper_bound)]
        mean_value = filtered_df[rim_station].mean()
        return mean_value

    def probabilities_average_flow(self, end_of_season_flow, mean_value, season, rim_stations):
        if season == "Kharif":
            probablity_table = self.probablity_table[rim_stations]
            end_of_Rabi = probablity_table.loc[0:17].sum()
            differences = abs(end_of_Rabi - mean_value)
            min_diff_column = differences.idxmin()
            prediction = np.array(probablity_table[min_diff_column][18:36])
            
        else:
            
            probablity_table = self.probablity_table[rim_stations]
            end_of_Kharif = probablity_table.loc[18:35].sum()
            differences = abs(end_of_Kharif - mean_value)
            min_diff_column = differences.idxmin()
            prediction = np.array(probablity_table[min_diff_column][0:18])
        return prediction


    def value(self, timestep, scenario_index):
        
        ts = self.model.timestepper.current
        start_year = self.model.timestepper.start.year
        days_in_month = timestep.period.days_in_month
        #there are two seasons Rabi (from April 1 to Sep 20) and Kharif (Oct to March)
        self.basin_wide_Rabi_prediction = []
        Rabi_start_time = pd.to_datetime(str(ts.year - 1) + '-09-29')
        Rabi_end_time = pd.to_datetime(str(ts.year) + '-03-30') 
        if ts.month == 9 and ts.day > 29:
            self.Rabi_season_start_date_index = ts.index

        #get_all_values
        #get_value
        if start_year < ts.year:
            if Rabi_start_time <= ts.datetime <= Rabi_end_time:
                pass

            if ts.month == 9 and ts.day > 29:
                self.Kharif_index = 0
                self.season = "Kharif"
                self.Kharif_prediction = {}
                for rim_station in self.rim_station:
                    self.node = self.node_[rim_station]
                    node_prev_flow = np.sum(self.node.data[ts.index - 180:ts.index])
                    rim_station_seasonal = rim_station + "_Kharif"
                    mean_value = self.calculate_matching_years(node_prev_flow, rim_station_seasonal)
                    self.prediction = list(self.probabilities_average_flow(node_prev_flow, mean_value, self.season, rim_station))
                    self.Kharif_prediction[rim_station] = self.prediction
                
                var_len = len(self.Kharif_prediction[list(self.Kharif_prediction.keys())[0]])
                self.basin_wide_Kharif_prediction = [sum([self.Kharif_prediction[rim_stat][index] for rim_stat in self.Kharif_prediction.keys()]) for index in range(var_len)]
                
            elif ts.month == 3 and ts.day > 30:
                self.Rabi_index = 0
                self.season = "Rabi"
                self.Rabi_prediction = {}
                for rim_station in self.rim_station:
                    self.node = self.node_[rim_station]
                    node_prev_flow = np.sum(self.node.data[ts.index - 180:ts.index])
                    rim_station_seasonal = rim_station + "_Rabi" 
                    mean_value = self.calculate_matching_years(node_prev_flow, rim_station_seasonal)
                    self.prediction = list(self.probabilities_average_flow(node_prev_flow, mean_value, self.season, rim_station))
                    self.Rabi_prediction[rim_station] = self.prediction
                var_len = len(self.Rabi_prediction[list(self.Rabi_prediction.keys())[0]])
                self.basin_wide_Rabi_prediction = [sum([self.Rabi_prediction[rim_stat][index] for rim_stat in self.Rabi_prediction.keys()]) for index in range(var_len)]
        
            
            #need to convert rim stations prediction to basin wide prediction
            #Update the predicted flow
            deficit_percentage_sindh = 0.552174606
            deficit_percentage_punjab = 0.447825394
            
            punjab_demand = sum([demand_node.get_max_flow(scenario_index) for demand_node in self.punjab_demand_nodes])
            sindh_demand = sum([demand_node.get_max_flow(scenario_index)  for demand_node in self.sindh_demand_nodes])


            if ts.month >= 4 and ts.month <= 9:
                #self.Rabi_index = "Rabi"
                total_Rabi_loss = 0
                Rabi_allocation_under_normal_condition = 2734261.2
                Rabi_punjab_share_under_normal_condition = 2734261.6*0.4 
                Rabi_sindh_share_under_normal_condition = 2734261.6*0.3 
                if ts.day == 10 or ts.day == 20:
                    observed_value = sum([sum(self.node_[inflow].data[ts.index - 10: ts.index]) for inflow in self.node_])
                    self.basin_wide_Rabi_prediction[self.Rabi_index] = observed_value 
                    self.Rabi_index += 1
                elif ts.day == days_in_month:
                    observed_value = sum([sum(self.node_[inflow].data[ts.index - days_in_month - 20: ts.index]) for inflow in self.node_])
                    self.basin_wide_Rabi_prediction[self.Rabi_index] = observed_value 
                    self.Rabi_index += 1
                Rabi_deficit_volume = (Rabi_allocation_under_normal_condition - (sum(self.basin_wide_Rabi_prediction) - total_Rabi_loss))

                if Rabi_deficit_volume > 0:
    
                    Rabi_punjab_deficit_fraction = (Rabi_deficit_volume*deficit_percentage_punjab/Rabi_punjab_share_under_normal_condition)
                    Rabi_sindh_deficit_fraction = (Rabi_deficit_volume*deficit_percentage_sindh/Rabi_sindh_share_under_normal_condition)
                    
                    Rabi_sindh_deficit_volume = sindh_demand*Rabi_sindh_deficit_fraction
                    Rabi_punjab_deficit_volume = punjab_demand*Rabi_punjab_deficit_fraction

                    for demand_node in self.punjab_demand_nodes:
                        demand_node.max_flow = (demand_node.get_max_flow(scenario_index)/punjab_demand)*Rabi_punjab_deficit_volume
                    for demand_node in self.sindh_demand_nodes:
                        demand_node.max_flow = (demand_node.get_max_flow(scenario_index)/sindh_demand)*Rabi_sindh_deficit_volume 


            elif ts.month < 4 and ts.month > 9:
                total_Kharif_loss = 0
                Kharif_allocation_under_normal_condition = 1514145.6
                Kharif_punjab_share_under_normal_condition = 1514145.6*0.4 
                Kharif_sindh_share_under_normal_condition = 1514145.6*0.3 
                if ts.day == 10 or ts.day == 20:
                    observed_value = sum([sum(self.node_[inflow].data[ts.index - 10: ts.index]) for inflow in self.node_])
                    self.basin_wide_Kharif_prediction[self.Kharif_index] = observed_value
                    self.Kharif_index += 1
                    
                elif ts.day == days_in_month:
                    observed_value = sum([sum(self.node_[inflow].data[ts.index - days_in_month - 20: ts.index]) for inflow in self.node_])
                    self.basin_wide_Kharif_prediction[self.Kharif_index] = observed_value
                    self.Kharif_index += 1
                Kharif_deficit_volume = (Kharif_allocation_under_normal_condition - (sum(self.basin_wide_Kharif_prediction) - total_Kharif_loss))

                if Kharif_deficit_volume > 0:

                    Kharif_punjab_deficit_fraction = (Kharif_deficit_volume*deficit_percentage_punjab/Kharif_punjab_share_under_normal_condition)
                    Kharif_sindh_deficit_fraction = (Kharif_deficit_volume*deficit_percentage_sindh/Kharif_sindh_share_under_normal_condition) 
                    
                    Kharif_sindh_deficit_volume = sindh_demand*Kharif_sindh_deficit_fraction
                    Kharif_punjab_deficit_volume = punjab_demand*Kharif_punjab_deficit_fraction

                    for demand_node in self.punjab_demand_nodes:
                        demand_node.max_flow = (demand_node.get_max_flow(scenario_index)/punjab_demand)*Kharif_punjab_deficit_volume
                    for demand_node in self.sindh_demand_nodes:
                        demand_node.max_flow = (demand_node.get_max_flow(scenario_index)/sindh_demand)*Kharif_sindh_deficit_volume 
        
        return 0


    @classmethod
    def load(cls, model, data):
        Historical_flow_Rim_station = data.pop("Historical_flow_Rim_station")
        IRSA_probablity_table = data.pop("IRSA_probablity_table")
        rim_station_observed = data.pop("rim_station_observed")
        reservoir_test = model._get_node_from_ref(model, data.pop("Reservoir_name"))
        
        node_ = {node : load_recorder(model, rim_station_observed[node]) for node in rim_station_observed}
        rim_station = [node for node in rim_station_observed]
        punjab_demand_nodes = [model._get_node_from_ref(model, node_name) for node_name in data["punjab_demand"]] 
        Punjab_channel_heads_node = {node_name:model._get_node_from_ref(model, node_name) for node_name in data.pop("punjab_demand")}
        sindh_demand_nodes = [model._get_node_from_ref(model, node_name) for node_name in data.pop("sindh_demand")]
        return cls(model, Punjab_channel_heads_node, reservoir_test, Historical_flow_Rim_station, IRSA_probablity_table, rim_station, node_, punjab_demand_nodes, sindh_demand_nodes, **data)
Rim_station_flow_forcast.register()

class Distribution_plan_parameter(Parameter):
    def __init__(self, model, distribution_plan, **kwargs):
        super().__init__(model, **kwargs)
        self.distribution_plan = distribution_plan 
        self.month_index = {1: "January",2: "February",3: "March",4: "April",5: "May",6: "June",7: "July",8: "August",9: "September",10: "October",11: "November",12: "December"}

    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        ts = self.model.timestepper.current
        month_name = self.month_index[ts.month] 

        if ts.day <= 10:
            key = month_name+"-1"
            return self.distribution_plan[key]
            
        elif ts.day > 10 and ts.day <= 20:
            key = month_name+"-2"
            return self.distribution_plan[key]
        else:
            key = month_name+"-3"
            return self.distribution_plan[key]
            
    @classmethod
    def load(cls, model, data):
        distribution_plan = data.pop("distribution_plan")
        return cls(model, distribution_plan, **data)      
Distribution_plan_parameter.register()

def water_balance_J_C_zone(input_data,Initial_Storage_Mangla,Maximum_Storage_Mangla,Initial_Storage_Tarbela,Maximum_Storage_Tarbela,Indus_at_Chashma,Storage_Dep_at_end_of_season_Mangla,Storage_Dep_at_end_of_Season_Tarbela,start_date,percentage_range,Filling_withdraw_fraction_Tarbela,Filling_withdraw_fraction_Mangla,Eastern_rivers,JC_average_system_uses_1977_1982,Average_System_use_Indus,KPK_Baloch_share,KPK_share_historical,Baloch_share_historical,Below_Kotri,Punjab_share_Indus_para_2_percent,System_losses_percent_Indus,System_losses_JC):
    #load input dataframe
    season = "Rabi"
    input_data = pd.HDFStore(input_data)
    observed_flow=input_data['/observed_flow']
    rim_station = ['Kabul_Noshehra', 'Indus_Tarbela', 'Jhelum_Mangla', 'Chenab_Marala']
    J_C_rim_stations_name = rim_station[2:4]
    Indus_rim_stations_name = rim_station[:2]
    observed_flow_index = observed_flow.index.get_loc(start_date)

    ##############################################Input_from_model############################################

    ##############################################Input_from_model############################################
    
    day_month_correction = [1, 1, 1.1, 1, 1, 1, 1, 1, 1.1, 1, 1, 1.1, 1, 1, 0.8, 1, 1, 1.1]
    CUMCS_MAC_conversion_factor = 0.01983471
    J_C_1977_88 = [49.6,45.7,42.5,39.1,37.1,35.9,36,34.4,25.7,12.7,14.6,20.9,27.7,33.2,29.5,32.9,35.7,35.6]
    J_C_1977_88_MAF = [CUMCS_MAC_conversion_factor*day_month_correction[index]*J_C_1977_88[index] for index in range(len(J_C_1977_88))]
 
    def convert_to_MAF(data):
        for station, data_dict in data.items():
            for key, arr in data_dict.items():
                data[station][key] = arr / 43560
        return data
    
    Rim_prediction = convert_to_MAF({rm : IRSA_prediction__Rabi(rm, input_data, season, observed_flow_index, percentage_range) for rm in rim_station})
    J_C_Rim_stations_prediction = {key: Rim_prediction[key] for key in J_C_rim_stations_name}
    Indus_Rim_stations_prediction = {key: Rim_prediction[key] for key in Indus_rim_stations_name}
    Min_JC = {}
    Max_JC = {}
    
    
    if season == "Rabi":
    
        J_C_Shortage = {}
        Punjab_J_C_Canal_Wdls_Outflow_RQBS = {}
        Sindh_Channel_dis_df = {}
        Punjab_J_C_Channel_dis_df = {}
        Punjab_Indus_Channel_dis_df = {}
        scenarios = ["Maximum", "Minimum"]
        for scenario in scenarios:
            
            ############################JJJJCCC################################################
            #Forecast of Inflows
            System_Outflow_RQBS_min_max = {}
            Jhelum_at_Mangla = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario])
            Chenab_at_Marala = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario])
            Rabi_Inflows_J_C_Command = Jhelum_at_Mangla + Chenab_at_Marala + sum(Eastern_rivers)
            
            
 
            Storage_Available = Initial_Storage_Mangla
            #Storage_Dep_at_end_of_Season_Mangla = 100
            Storage_Release = Storage_Available - (100 - Storage_Dep_at_end_of_season_Mangla)*Maximum_Storage_Mangla
            System_Inflows = Storage_Release + Rabi_Inflows_J_C_Command
            Total_Availability_JC = System_Inflows * (1 - System_losses_JC/100)
            
            ##############################IIINNNN#################################################

            Indus_at_Tarbela = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario])
            Kabul_at_Nowshehra = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario])
            Total_Indus = Indus_at_Tarbela + Kabul_at_Nowshehra + Indus_at_Chashma
            
            Storage_Available_Tarbela = Initial_Storage_Tarbela
            Storage_Release_Tarbela = Initial_Storage_Tarbela - (100 - Storage_Dep_at_end_of_Season_Tarbela)*Maximum_Storage_Tarbela
            KPK_share = [day_month_correction[index]*KPK_share_historical[index]*0.01983471 for index in range(len(KPK_share_historical))]
            Baloch_share = [day_month_correction[index]*Baloch_share_historical[index]*0.01983471 for index in range(len(KPK_share_historical))]
            System_losses_Indus = 1 - System_losses_percent_Indus/100
            
            
            Indus_know_parameters = System_losses_Indus*(Storage_Release_Tarbela + Total_Indus) - KPK_Baloch_share - Below_Kotri
            Numerator = Total_Availability_JC*Average_System_use_Indus - JC_average_system_uses_1977_1982*Indus_know_parameters
            Denominator = System_losses_Indus*JC_average_system_uses_1977_1982 + Average_System_use_Indus
            J_C_Outflow = Numerator/Denominator
            
            """
            System_Inflows_Indus = Total_Indus + sum(J_C_Outflow) + Storage_Release_Tarbela
            System_losses_Indus = System_losses_percent_Indus*System_Inflows_Indus/100
            """
            
            ############################JJJJCCC################################################
            JC_Canal_Availability = Total_Availability_JC - J_C_Outflow  #Where is this value comming from?
            J_C_Shortage = (1 - JC_Canal_Availability/JC_average_system_uses_1977_1982)*100
    
            ############################JJJJCCC################################################
            Live_content_MAF = [Initial_Storage_Mangla]  #in MAC
            Initial_Storage_temp = Initial_Storage_Mangla
            
            for vol in Filling_withdraw_fraction_Mangla:
                if Initial_Storage_temp - Storage_Release*vol/100 < 0:
                    Live_content_MAF.append(0)
                else:
                    release = (Storage_Release*vol/100)
                    Initial_Storage_temp = round(Initial_Storage_temp - release, 3)
                    Live_content_MAF.append(Initial_Storage_temp)
            
            
            Mangla_outflow = []
            for index in range(18):
                release = J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario][index] - (Live_content_MAF[index+1]-Live_content_MAF[index])
                
                Mangla_outflow.append(release)

            System_Inflow = [Mangla_outflow[index] + Eastern_rivers[index] + J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario][index] for index in range(len(Mangla_outflow))]
            Net_inflow_after_loss = [-1*inflow*(System_losses_JC/100) for inflow in System_Inflow]
            Net_inflow = [System_Inflow[index] - Net_inflow_after_loss[index] for index in range(len(Net_inflow_after_loss))]
            
            Punjab_J_C_Canal_Wdls = []
            Punjab_J_C_Canal_Wdls = [
                Net_inflow[index] if (1 - J_C_Shortage * 0.01) * J_C_1977_88_MAF[index] > Net_inflow[index] 
                else (1 - J_C_Shortage * 0.01) * J_C_1977_88_MAF[index]
                for index in range(len(Net_inflow))
            ]
            System_Outflow_RQBS = [Net_inflow[index] - Punjab_J_C_Canal_Wdls[index] for index in range(len(Punjab_J_C_Canal_Wdls))]

            
            
            ############################JJJJCCC################################################

            ##############################IIINNNN#################################################
            #JC_outflow is part of the below equation that need to be fixed
            
            #JC_outflow is part of the below equation that need to be fixed
            System_Inflows_Indus = Total_Indus + J_C_Outflow + Storage_Release_Tarbela
            System_losses_Indus_volume = System_Inflows_Indus*System_losses_percent_Indus/100
            
            Total_Availability_Indus = System_Inflows_Indus - System_losses_Indus_volume
            Canal_Availability_Indus = Total_Availability_Indus - Below_Kotri
            Punjab_Sindh_share_Indus = Canal_Availability_Indus - KPK_Baloch_share
            Shortage = (1 - Punjab_Sindh_share_Indus/Average_System_use_Indus)*100
            
            ##############################IIINNNN#################################################
        
            
            
            ##############################IIINNNN#######################################################################
            Live_content_MAF = [Initial_Storage_Tarbela]  #in MAC
            Initial_Storage_temp = Initial_Storage_Tarbela

            for vol in Filling_withdraw_fraction_Tarbela:
                if Initial_Storage_temp - Storage_Release_Tarbela*vol/100 < 0:
                    Live_content_MAF.append(0)
                else:
                    release = (Storage_Release_Tarbela*vol/100)
                    Initial_Storage_temp = round(Initial_Storage_temp - release, 3)
                    Live_content_MAF.append(Initial_Storage_temp)
                                        
            Tarbela_outflow = []
            for index in range(len(Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario])):
                release = Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario][index]-(Live_content_MAF[index+1]-Live_content_MAF[index])
                Tarbela_outflow.append(release)
            System_Inflow = [Tarbela_outflow[index] + Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario][index] + System_Outflow_RQBS[index] for index in range(len(Tarbela_outflow))]
            Net_inflow_after_loss_Indus = [-1*inflow*System_losses_percent_Indus/100 for inflow in System_Inflow]
            Net_inflow_Indus = [System_Inflow[x] + Net_inflow_after_loss_Indus[x] for x in range(len(Net_inflow_after_loss_Indus))]
            
            Proposed_Canal_Wdls_Indus = Net_inflow_Indus    
            System_Outflow_DS_Kotri = [Net_inflow_Indus[index]-Proposed_Canal_Wdls_Indus[index] for index in range(len(Net_inflow_Indus))]

            
            
            Total_Share_Punjab_Sindh = [Proposed_Canal_Wdls_Indus[index] - KPK_share[index] - Baloch_share[index] for index in range(len(KPK_share))]
            #condition to implement the Para2 para14b and 
            Sindh_share_Indus_para_2_percent = [100 - Punjab_share_Indus_para_2_percent[index] for index in range(len(Punjab_share_Indus_para_2_percent))]
            
            
            Punjab_Indus_Canal_Wdls = [Total_Share_Punjab_Sindh[index]*Punjab_share_Indus_para_2_percent[index]*0.01 for index in range(len(Punjab_share_Indus_para_2_percent))]
            Sindh_share_Canal_Wdls  = [Total_Share_Punjab_Sindh[index]*Sindh_share_Indus_para_2_percent[index]*0.01 for index in range(len(Punjab_share_Indus_para_2_percent))]

            ##############################IIINNNN#######################################################################
            
            Sindh_Channel_dis_df[scenario] = input_data['Sindh_Channel_dis_plan_percentage'][0:18].mul(Sindh_share_Canal_Wdls, axis=0)
            Punjab_J_C_Channel_dis_df[scenario] = input_data['J_C_Channel_dis_plan_percentage'][0:18].mul(Punjab_J_C_Canal_Wdls, axis=0)
            Punjab_Indus_Channel_dis_df[scenario] = input_data['Indus_Channel_dis_plan_percentage'][0:18].mul(Punjab_Indus_Canal_Wdls, axis=0)
            System_Outflow_RQBS_min_max[scenario] = System_Outflow_RQBS 

            
        output = {}
        Sindh_Channel_dis_df_likely = (Sindh_Channel_dis_df[scenarios[0]] + Sindh_Channel_dis_df[scenarios[1]]) / 2
        Punjab_J_C_Channel_dis_df_likely = (Punjab_J_C_Channel_dis_df[scenarios[0]] + Punjab_J_C_Channel_dis_df[scenarios[1]]) / 2
        Punjab_Indus_Channel_dis_df_likely = (Punjab_Indus_Channel_dis_df[scenarios[0]] + Punjab_Indus_Channel_dis_df[scenarios[1]]) / 2
        RQBS_Canal_Outflow_likely =  [sum(x) / len(x) for x in zip(*System_Outflow_RQBS_min_max.values())]
        
 
        output["Sindh_Channel_dis_df_likely"] = Sindh_Channel_dis_df_likely
        output["Punjab_J_C_Channel_dis_df_likely"] = Punjab_J_C_Channel_dis_df_likely
        output["Punjab_Indus_Channel_dis_df_likely"] = Punjab_Indus_Channel_dis_df_likely
        output["RQBS_Canal_Outflow_likely"] = RQBS_Canal_Outflow_likely
        
            

        #RQBS_Canal_Outflow_likely.to_csv("RQBS_Canal_Outflow_likely")
    elif season == "Khrif":
        #ToDo Khrif implementation
        pass
    
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ImplementationPlan@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    """
    1. update the IRSA prediction with observed flow and reservoir active Storage
    2. redo the calculation to get how much water is avaiable
    3. check if there is enough water to satisfy the provinital water allocaiton and make sure that the provinces does not use above the allocated water 
    4. curtail the demand if the available water is below the remaining demand (know which to curtail)
    """
    
    """
    forcing the model to release the RQBS flow to Indus basin.
    """
    input_data.close()
    return output

def IRSA_prediction__Rabi(rim_station_name, input_df, season, index, percentage_range):
    
    observed_flow_values = input_df['/observed_flow'][rim_station_name]
    probablity_table = input_df[rim_station_name]
    Flow_rim_station = input_df['/Historical_flow']
    probablity_column = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    def calculate_matching_years(node_prev_flow, rim_station):
        lower_bound = node_prev_flow * (1 - percentage_range[rim_station_name])
        upper_bound = node_prev_flow * (1 + percentage_range[rim_station_name])
        filtered_df = Flow_rim_station[(Flow_rim_station[rim_station] >= lower_bound) & (Flow_rim_station[rim_station] <= upper_bound)]
        
        mean_value = filtered_df[rim_station].mean()
        return mean_value

    def probabilities_average_flow(end_of_season_flow, mean_value, season):
        min_max = {'Minimum': -0.1, 'Maximum': 0.1}
        prediction_min_max = {}
        if season == "Kharif":
            end_of_Rabi = probablity_table.loc[18:36].sum()
            differences = abs(end_of_Rabi - mean_value)
            
            for key in min_max:
                val = min_max[key]
                if differences.idxmin() == "SYN Max":
                    probablity_val = 0.05
                elif differences.idxmin() == "SYN Min":
                    probablity_val = 0.95
                else:
                    probablity_val = differences.idxmin()
                
                min_diff_column = probablity_val + val
                closest_number = min(probablity_column, key=lambda x: abs(x - min_diff_column))
                prediction = np.array(probablity_table[closest_number][0:18])
                prediction_min_max[key] = prediction

        else:
            end_of_Kharif = probablity_table.loc[0:18].sum()
            differences = abs(end_of_Kharif - mean_value)
            
            for key in min_max:
                val = min_max[key]
                if differences.idxmin() == "SYN Max":
                    probablity_val = 0.05
                elif differences.idxmin() == "SYN Min":
                    probablity_val = 0.95
                else:
                    probablity_val = differences.idxmin()
                
                min_diff_column = probablity_val + val
                closest_number = min(probablity_column, key=lambda x: abs(x - min_diff_column))
                prediction = np.array(probablity_table[closest_number][18:36])
                prediction_min_max[key] = prediction
                
        return prediction_min_max
    

    if season == "Kharif":
        node_prev_flow = np.sum(observed_flow_values[index - 183 : index])
        #rim_station = rim_station_name
        rim_station_seasonal = rim_station_name + "_Rabi" 
        mean_value = calculate_matching_years(node_prev_flow, rim_station_seasonal)
        
        prediction = probabilities_average_flow(node_prev_flow, mean_value, season)

    else:
        node_prev_flow = np.sum(observed_flow_values[index - 183 : index])
        rim_station_seasonal = rim_station_name + "_Kharif" 
        mean_value = calculate_matching_years(node_prev_flow, rim_station_seasonal)
        prediction = probabilities_average_flow(node_prev_flow, mean_value, season)
    return prediction

def IRSA_prediction__Kharif(rim_station, input_data, season, current_index, percentage_tolerance):
    """
    Predicts the flow for a given rim station and season based on historical flow data and probability tables.
    
    Parameters:
    -----------
    rim_station : str
        The name of the rim station for which the prediction is made.
        
    input_data : pd.DataFrame
        A DataFrame containing observed flow values, historical flows, and probability tables.
        
    season : str
        The season for which to predict the flow ("Kharif" or "Rabi").
        
    current_index : int
        The index representing the current day or period in the observed flow data.
        
    percentage_tolerance : float
        The percentage range for tolerance when finding matching years for flow interpolation.
        
    Returns:
    --------
    dict
        A dictionary containing flow predictions for "Minimum" and "Maximum" probability scenarios for the given season.
    """
    
    # Extract necessary data from the input dataframe
    percentage_tolerance = percentage_tolerance[rim_station]
    observed_flows = input_data['/observed_flow'][rim_station]
    probability_table = input_data[rim_station]
    historical_flow_data = input_data['/Historical_flow_new']
    probability_intervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def find_matching_years(previous_flow, seasonal_rim_station):
        """
        Finds the historical years where the flow is within a specified tolerance of the previous season flow.
        
        Parameters:
        -----------
        previous_flow : float
            The calculated flow from the previous season or time period.
            
        seasonal_rim_station : str
            The name of the rim station for the given season.
            
        Returns:
        --------
        float
            The mean flow value from the matching historical years.
        """
        lower_bound = previous_flow * (1 - percentage_tolerance)
        upper_bound = previous_flow * (1 + percentage_tolerance)
        
        matching_flow_data = historical_flow_data[(historical_flow_data[seasonal_rim_station] >= lower_bound) & 
                                                  (historical_flow_data[seasonal_rim_station] <= upper_bound)]
        
        if rim_station + "_Rabi" == seasonal_rim_station:
            mean_flow = {}
            mean_flow["Early_Kharif"] = matching_flow_data[rim_station + "_Early_Kharif"].mean()
            mean_flow["Late_Kharif"] = matching_flow_data[rim_station + "_Late_Kharif"].mean()
            
        return mean_flow
    def calculate_seasonal_predictions(mean_flow, season):
        """
        Calculates the flow predictions based on probability tables and differences from the mean flow.

        Parameters:
        -----------
        mean_flow : float or dict
            The average flow from historical data matching the previous flow.

        season : str
            The season for which the prediction is being made ("Kharif" or "Rabi").

        Returns:
        --------
        dict
            A dictionary containing flow predictions for "Minimum" and "Maximum" scenarios.
        """
        def get_probability_val(differences):
            if differences.idxmin() == "SYN Max":
                return 0.05
            elif differences.idxmin() == "SYN Min":
                return 0.95
            return differences.idxmin()

        def make_prediction(differences, period, adjustment):
            probability_val = get_probability_val(differences)
            adjusted_probability = probability_val + adjustment
            closest_probability = min(probability_intervals, key=lambda x: abs(x - adjusted_probability))
            return np.array(probability_table[closest_probability][period])

        prediction_results = {}
        min_max_scenarios = {'Minimum': -0.1, 'Maximum': 0.1}

        if season == "Kharif":
            EK_season, LK_season = probability_table.loc[:7].sum(), probability_table.loc[7:18].sum()
            differences_EK = abs(EK_season - mean_flow["Early_Kharif"])
            differences_LK = abs(LK_season - mean_flow["Late_Kharif"])

            for scenario, adjustment in min_max_scenarios.items():
                prediction_results[scenario] = {
                    "Early_Kharif": make_prediction(differences_EK, slice(0, 7), adjustment),
                    "Late_Kharif": make_prediction(differences_LK, slice(7, 18), adjustment)
                }
        else:
            end_of_kharif_season = probability_table.loc[18:36].sum()
            differences = abs(end_of_kharif_season - mean_flow)

            for scenario, adjustment in min_max_scenarios.items():
                prediction_results[scenario] = make_prediction(differences, slice(18, 36), adjustment)

        return prediction_results

    # Calculate previous season flow for the given rim station
    seasonal_flow_sum = np.sum(observed_flows[current_index - 183 : current_index])
    seasonal_rim_station = rim_station + "_Rabi" 

    # Find the matching years and calculate predictions
    average_flow = find_matching_years(seasonal_flow_sum, seasonal_rim_station)
    flow_predictions = calculate_seasonal_predictions(average_flow, season)

    return flow_predictions

class Rabi_water_allocation__(Parameter):
    def __init__(self, model, Punjab_channel_heads, Sindh_channel_heads, 
    input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, 
    Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, 
    percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, 
    Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, 
    KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, 
    System_losses_percent_Indus, System_losses_JC, **kwargs):
        
        super().__init__(model, **kwargs)
        self.input_data = input_data
        self.Mangla_reservoir = Mangla_reservoir
        self.Tarbela_reservoir = Tarbela_reservoir
        
        self.Punjab_channel_heads = Punjab_channel_heads
        self.Sindh_channel_heads = Sindh_channel_heads
        
        self.Punjab_channel_heads_node = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Punjab_channel_heads}

        self.Sindh_channel_heads_Guddu = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Guddu"]}
        self.Sindh_channel_heads_Sukkur = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Sukkur"]}
        self.Sindh_channel_heads_Kotri = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Kotri"]}


        self.Sindh_channel_heads_node = {**self.Sindh_channel_heads_Guddu, **self.Sindh_channel_heads_Sukkur, **self.Sindh_channel_heads_Kotri}
        self.Punjab_channel_heads_recorders_name = [node+"_rec" for node in self.Punjab_channel_heads]
        self.Sindh_channel_heads_recorders_name = [node+"_rec" for node in self.Sindh_channel_heads["Guddu"] + self.Sindh_channel_heads["Sukkur"] + self.Sindh_channel_heads["Kotri"]]

        self.Punjab_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Punjab_channel_heads_recorders_name}
        self.Sindh_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Sindh_channel_heads_recorders_name}
        
        self.Indus_at_Chashma = Indus_at_Chashma
        self.Storage_Dep_at_end_of_season_Mangla = Storage_Dep_at_end_of_season_Mangla
        self.Storage_Dep_at_end_of_Season_Tarbela = Storage_Dep_at_end_of_Season_Tarbela
        self.percentage_range = percentage_range
        self.Filling_withdraw_fraction_Tarbela = Filling_withdraw_fraction_Tarbela
        self.Filling_withdraw_fraction_Mangla = Filling_withdraw_fraction_Mangla
        
        self.Eastern_rivers = Eastern_rivers
        self.JC_average_system_uses_1977_1982 = JC_average_system_uses_1977_1982
        self.Average_System_use_Indus = Average_System_use_Indus
        self.KPK_Baloch_share = KPK_Baloch_share
        self.KPK_share_historical = KPK_share_historical
        self.Baloch_share_historical = Baloch_share_historical
        
        self.Below_Kotri = Below_Kotri
        self.Punjab_share_Indus_para_2_percent = Punjab_share_Indus_para_2_percent
        self.System_losses_percent_Indus = System_losses_percent_Indus
        self.System_losses_JC = System_losses_JC

    def setup(self):
        self.channel_head_to_recorder_name = {
            'LJC': 'LJC_total_rec',
            'UJC_INT': 'UJC_int_rec',
            'LCC': 'LCC_rec',
            'UCC_INT': 'UCC_canal_rec',
            'LBDC': 'LBDC_before_MP_link_rec',
            'M.R_INT': 'MR_link_int_rec',

            'UDC': 'UDC_rec',
            'LDC': 'LDC_rec',
            'UPC': 'UPC_rec',
            'FC': 'fordwah_canal_rec',

            'CRBC_PB':'crbc_kpk_canal_rec', 
            'HAVELI_INT': 'havali_int_rec',
            'SIDHNAI': 'sidhnai_canal_rec', 
            'RANGPUR': 'rangpur_canal_rec',
            'PANJNAD': 'panjnad_canal_rec',
            'ABBASIA LINK': 'abbasia_link_canal_rec',

            'THAL': 'Thal Canal_rec' ,  
            'CBDC': 'CBDC_rec',
            'UBC+QC': ['qaim_canal_rec', 'UBC_rec'],
            'LBC': 'LBC_rec',
            'MZG': 'muzafargarh_canal_rec',
            'DG_KHAN': 'DGK_canal_rec'
            }
        super().setup()

    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        ts = self.model.timestepper.current
        days_in_month = timestep.period.days_in_month
        start_date = str(ts.year)+"-"+str(ts.month)+"-"+str(ts.day)
        self.val = 0
        start_year = self.model.timestepper.start.year
        Initial_Storage_Mangla = self.Mangla_reservoir.volume[i]/43560
        Maximum_Storage_Mangla = self.Mangla_reservoir.max_volume/43560
        Initial_Storage_Tarbela = self.Tarbela_reservoir.volume[i]/43560
        Maximum_Storage_Tarbela = 314166/43560
        
        self.pubjab_abstructed_Rabi = {}
        self.sindh_abstructed_Rabi = {}
        
        self.pubjab_remaining_demand_Rabi = {}
        self.sindh_remaining_demand_Rabi = {}
        
        if ts.month == 9 and ts.day > 29:
            self.Rabi_season_start_date_index = ts.index
        
        if start_year < ts.year:
            if ts.month == 9 and ts.day > 29:
                self.Rabi_index = 0
                self.Punjab_Sindh_channel_heads_index = 0
                
                self.Rabi_season_start_date_index = ts.index
                water_balance_J_C_zone_output = water_balance_J_C_zone(self.input_data, Initial_Storage_Mangla, Maximum_Storage_Mangla, Initial_Storage_Tarbela, Maximum_Storage_Tarbela, self.Indus_at_Chashma, self.Storage_Dep_at_end_of_season_Mangla, self.Storage_Dep_at_end_of_Season_Tarbela, start_date, self.percentage_range, self.Filling_withdraw_fraction_Tarbela, self.Filling_withdraw_fraction_Mangla, self.Eastern_rivers, self.JC_average_system_uses_1977_1982, self.Average_System_use_Indus, self.KPK_Baloch_share, self.KPK_share_historical, self.Baloch_share_historical, self.Below_Kotri, self.Punjab_share_Indus_para_2_percent, self.System_losses_percent_Indus, self.System_losses_JC) 
                
                self.Sindh_Channel_dis_df_likely = water_balance_J_C_zone_output["Sindh_Channel_dis_df_likely"]
                self.Punjab_J_C_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_J_C_Channel_dis_df_likely"]
                self.Punjab_Indus_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_Indus_Channel_dis_df_likely"]
                self.RQBS_Canal_Outflow_likely = water_balance_J_C_zone_output["RQBS_Canal_Outflow_likely"]
                
                self.Sindh_total_allocated_water = self.Sindh_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_J_C_total_allocated_water = self.Punjab_J_C_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_Indus_total_allocated_water = self.Punjab_Indus_Channel_dis_df_likely.sum(axis=1).tolist()
            # If the current date is in the first quarter of the year, the Rabi season started last year
            if ts.month <= 3:  # January, February, March
                self.Rabi_start_time = pd.Timestamp(f'{ts.year-1}-10-01')
                self.Rabi_end_time = pd.Timestamp(f'{ts.year}-03-30')
            else:
                # Otherwise, the Rabi season is in the current year and the next year
                self.Rabi_start_time = pd.Timestamp(f'{ts.year}-10-01')
                self.Rabi_end_time = pd.Timestamp(f'{ts.year+1}-03-30')


            if self.Rabi_start_time <= ts.datetime <= self.Rabi_end_time and ts.datetime >  pd.Timestamp('2013-10-01'):
                
                #water used sofar
                self.pubjab_abstructed_Rabi = {recorder : sum(self.Punjab_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Punjab_channel_heads_recorders}
                self.sindh_abstructed_Rabi = {recorder : sum(self.Sindh_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Sindh_channel_heads_recorders}
                
                #remining water that is required to satisfy the full demand
                self.pubjab_remaining_demand_Rabi = {node : sum(self.Punjab_channel_heads_node[node].max_flow.dataframe[ts.datetime : self.Rabi_end_time].values) for node in self.Punjab_channel_heads_node}
                self.sindh_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_node[node].max_flow.dataframe[ts.datetime : self.Rabi_end_time].values) for node in self.Sindh_channel_heads_node}
                self.Guddu_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_Guddu[node].max_flow.dataframe[ts.datetime : self.Rabi_end_time].values) for node in self.Sindh_channel_heads_Guddu}
                self.Sukkur_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_Sukkur[node].max_flow.dataframe[ts.datetime : self.Rabi_end_time].values) for node in self.Sindh_channel_heads_Sukkur}
                self.Kotri_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_Kotri[node].max_flow.dataframe[ts.datetime : self.Rabi_end_time].values) for node in self.Sindh_channel_heads_Kotri}
                
                #how much avaiable is avaiable to allocate according to IRSA's prediction 
                if ts.day == 10 or ts.day == 20:
                    self.Sindh_Channel_available_water_to_allocate = sum(self.Sindh_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    
                    self.Punjab_J_C_Channel_available_water_to_allocate = sum(self.Punjab_J_C_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_Indus_Channel_available_water_to_allocate = sum(self.Punjab_Indus_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    
                    
                    #check if the aggregated required water demand is greather aggregated available water
                    if sum(self.pubjab_remaining_demand_Rabi.values()) > self.Punjab_J_C_Channel_available_water_to_allocate + self.Punjab_Indus_Channel_available_water_to_allocate:
                        #calculate which channel head to curtail and by how much
                        Punjab_channel_heads = list(self.Punjab_J_C_Channel_dis_df_likely.columns) + list(self.Punjab_Indus_Channel_dis_df_likely.columns)
                        for channel_head in Punjab_channel_heads:
                            try:
                                whats_should_be_allocated = self.Punjab_J_C_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum()
                            except KeyError:
                                whats_should_be_allocated = self.Punjab_Indus_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum()
                            if channel_head not in  ['ESC', 'LPC', 'LMC', 'CBDC', 'UBC+QC']:
                                channel_head_node_name = self.channel_head_to_recorder_name[channel_head][:-4]
                                what_is_allocated = self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head]]
                                if what_is_allocated > whats_should_be_allocated:
                                    curtailment = 0.01
                                    #self.Punjab_channel_heads_node[channel_head_node_name].max_flow = self.Punjab_channel_heads_node[channel_head_node_name].get_max_flow(scenario_index)*(1- curtailment)
                            if channel_head == 'UBC+QC':
                                what_is_allocated = self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head][0]][0] + self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head][1]][0]
                                if what_is_allocated > whats_should_be_allocated:
                                    curtailment = 0.01 
                            
                    if sum(self.sindh_remaining_demand_Rabi.values()) > self.Sindh_Channel_available_water_to_allocate:
                        #sum(self.Punjab_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index])
                        self.Guddu_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Guddu}
                        self.Sukkur_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Sukkur}
                        self.Kotri_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Kotri}
                        
                        for channel_head in list(self.Sindh_Channel_dis_df_likely.columns):
                            whats_should_be_allocated = (self.Sindh_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum())*43560
                            if channel_head == "Guddu":
                                if sum(self.Guddu_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Guddu_remaining_demand_Rabi.values()) 
                            elif channel_head == "Sukkur":
                                if sum(self.Sukkur_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Sukkur_remaining_demand_Rabi.values()) 
                            else:
                                if sum(self.Kotri_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Kotri_remaining_demand_Rabi.values())

                    self.Rabi_index += 1
                
                elif ts.day == days_in_month:
                    self.Sindh_Channel_available_water_to_allocate = sum(self.Sindh_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_J_C_Channel_available_water_to_allocate = sum(self.Punjab_J_C_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_Indus_Channel_available_water_to_allocate = sum(self.Punjab_Indus_Channel_dis_df_likely[self.Rabi_index:].sum())*43560

                    #check if the aggregated required water demand is greather aggregated available water
                    if sum(self.pubjab_remaining_demand_Rabi.values()) > self.Punjab_J_C_Channel_available_water_to_allocate + self.Punjab_Indus_Channel_available_water_to_allocate:
                        #calculate which channel head to curtail and by how much
                        Punjab_channel_heads = list(self.Punjab_J_C_Channel_dis_df_likely.columns) + list(self.Punjab_Indus_Channel_dis_df_likely.columns)
                        for channel_head in Punjab_channel_heads:
                            try:
                                whats_should_be_allocated = self.Punjab_J_C_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum()
                            except KeyError:
                                whats_should_be_allocated = self.Punjab_Indus_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum()
                            if channel_head not in  ['ESC', 'LPC', 'LMC', 'CBDC', 'UBC+QC']:
                                channel_head_node_name = self.channel_head_to_recorder_name[channel_head][:-4]
                                what_is_allocated = self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head]]
                                if what_is_allocated > whats_should_be_allocated:
                                    curtailment = 0.01
                                    #self.Punjab_channel_heads_node[channel_head_node_name].max_flow = self.Punjab_channel_heads_node[channel_head_node_name].get_max_flow(scenario_index)*(1- curtailment)
                            if channel_head == 'UBC+QC':
                                what_is_allocated = self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head][0]] + self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head][1]]
                                if what_is_allocated > whats_should_be_allocated:
                                    curtailment = 0.01 
                            
                    if sum(self.sindh_remaining_demand_Rabi.values()) > self.Sindh_Channel_available_water_to_allocate:
                        #sum(self.Punjab_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index])
                        self.Guddu_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Guddu}
                        self.Sukkur_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Sukkur}
                        self.Kotri_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Kotri}
                        
                        for channel_head in list(self.Sindh_Channel_dis_df_likely.columns):
                            whats_should_be_allocated = (self.Sindh_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum())*43560
                            if channel_head == "Guddu":
                                if sum(self.Guddu_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Guddu_remaining_demand_Rabi.values()) 
                            elif channel_head == "Sukkur":
                                if sum(self.Sukkur_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Sukkur_remaining_demand_Rabi.values()) 
                            else:
                                if sum(self.Kotri_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Kotri_remaining_demand_Rabi.values())
                    self.Rabi_index += 1
                    
        else:
            self.track_year = ts.year
        val = sum(self.pubjab_remaining_demand_Rabi.values())
        return val
            
    @classmethod
    def load(cls, model, data):
        Mangla_reservoir = model._get_node_from_ref(model, data.pop("Mangla_reservoir_node"))
        Tarbela_reservoir = model._get_node_from_ref(model, data.pop("Tarbela_reservoir_node"))
        
        Indus_at_Chashma = data.pop("Indus_at_Chashma")
        Storage_Dep_at_end_of_season_Mangla = data.pop("Storage_Dep_at_end_of_season_Mangla")
        Storage_Dep_at_end_of_Season_Tarbela = data.pop("Storage_Dep_at_end_of_Season_Tarbela")
        System_losses = data.pop("System_losses")
        
        percentage_range = data.pop("percentage_range")
        Filling_withdraw_fraction_Tarbela = data.pop("Filling_withdraw_fraction_Tarbela")
        Filling_withdraw_fraction_Mangla = data.pop("Filling_withdraw_fraction_Mangla")
        Eastern_rivers = data.pop("Eastern_rivers")
        
        JC_average_system_uses_1977_1982 = data.pop("JC_average_system_uses_1977_1982")
        Average_System_use_Indus = data.pop("Average_System_use_Indus")
        KPK_Baloch_share = data.pop("KPK_Baloch_share")
        KPK_share_historical = data.pop("KPK_share_historical")
        
        Baloch_share_historical = data.pop("Baloch_share_historical")
        Below_Kotri = data.pop("Below_Kotri")
        Punjab_share_Indus_para_2_percent = data.pop("Punjab_share_Indus_para_2_percent")
        System_losses_percent_Indus = data.pop("System_losses_percent_Indus")
        System_losses_JC = data.pop("System_losses_JC")
        
        input_data = data.pop("url")
        Punjab_channel_heads = data.pop("Punjab_channel_heads")
        Sindh_channel_heads = data.pop("Sindh_channel_heads")
        
        return cls(model, Punjab_channel_heads, Sindh_channel_heads, input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, System_losses_percent_Indus, System_losses_JC, **data)
Rabi_water_allocation__.register()


def Kharif_Indus_J_C_zone_water_balance(KPK_Baloch_share_Early_Kharif, KPK_Baloch_share_Late_Kharif, Below_Kotri_Early_Kharif, Below_Kotri_Late_Kharif, Indus_Early_Kharif_loss_percent, 
Indus_Late_Kharif_loss_percent, J_C_Early_Kharif_loss, 
J_C_Late_Kharif_loss, Storage_to_fill_in_E_Kharif_Tarbela, Storage_to_fill_in_L_Kharif_Tarbela, Storage_to_fill_in_E_Kharif_Mangla, Storage_to_fill_in_L_Kharif_Mangla, input_data,Initial_Storage_Mangla,Maximum_Storage_Mangla,
Initial_Storage_Tarbela,Maximum_Storage_Tarbela,
Indus_at_Chashma,Storage_Dep_at_end_of_season_Mangla,Storage_Dep_at_end_of_Season_Tarbela,start_date,percentage_range,
Filling_withdraw_fraction_Tarbela,Filling_withdraw_fraction_Mangla,Eastern_rivers,
JC_average_system_uses_1977_1982,Average_System_use_Indus,
KPK_Baloch_share,KPK_share_historical,Baloch_share_historical,Below_Kotri,
Punjab_share_Indus_para_2_percent,System_losses_percent_Indus,System_losses_JC):
    

    # Define the shortage percentages and ten-day index
    shortage_percentages = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ten_day_indices = np.arange(1, 19)  # Assuming consecutive numbering

    # Define the table values (you need to replace this with actual values from your dataset)
    Kharif_shortage_look_up_table = np.array([
        [34.4, 29.5, 24.0, 24.0, 24.0, 24.0, 24.0],
        [37.6, 30.4, 26.0, 26.0, 26.0, 26.0, 26.0],
        [44.5, 40.0, 33.2, 33.2, 33.2, 33.2, 33.2],
        [49.5, 49.5, 49.5, 32.9, 32.9, 32.9, 32.9],
        [52.1, 52.1, 52.1, 52.1, 35.4, 35.4, 35.4],
        [54.3, 54.3, 54.3, 54.3, 54.3, 39.0, 39.0],
        [55.7, 55.7, 55.7, 55.7, 55.7, 55.7, 39.0],
        [57.7, 57.7, 57.7, 57.7, 36.0, 30.0, 30.0],
        [59.4, 59.4, 59.4, 59.4, 59.4, 59.4, 49.0],
        [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 49.0],
        [54.5, 54.5, 54.5, 54.5, 54.5, 54.5, 49.0],
        [52.5, 52.5, 52.5, 52.5, 52.5, 52.5, 50.0],
        [53.1, 53.1, 53.1, 53.1, 53.1, 53.1, 50.0],
        [58.7, 58.7, 58.7, 58.7, 58.7, 48.7, 48.7],
        [62.1, 62.1, 62.1, 38.0, 34.0, 30.0, 30.0],
        [61.4, 53.5, 41.0, 37.0, 34.0, 30.0, 30.0],
        [60.3, 48.0, 39.0, 37.0, 34.0, 30.0, 30.0],
        [57.2, 45.0, 34.0, 34.0, 34.0, 30.0, 30.0]
    ])
    interpolator = scipy.interpolate.RegularGridInterpolator((ten_day_indices, shortage_percentages), Kharif_shortage_look_up_table)
    def interpolate_value(shortage, ten_day):
        #print(ten_day,shortage)
        return interpolator([[ten_day, shortage]])[0]

    season = "Kharif"
    input_data = pd.HDFStore(input_data)
    observed_flow=input_data['/observed_flow']
    rim_station = ['Kabul_Noshehra', 'Indus_Tarbela', 'Jhelum_Mangla', 'Chenab_Marala']
    J_C_rim_stations_name = rim_station[2:4]
    Indus_rim_stations_name = rim_station[:2]
    observed_flow_index = observed_flow.index.get_loc(start_date)

    day_month_correction = [1, 1, 1.1, 1, 1, 1, 1, 1, 1.1, 1, 1, 1.1, 1, 1, 0.8, 1, 1, 1.1]
    CUMCS_MAC_conversion_factor = 0.01983471
    J_C_1977_88 = [34.4, 37.6, 44.5, 49.5, 52.1, 54.3, 55.7, 57.7, 59.4, 59.5, 54.5, 52.5, 53.1, 58.7, 62.1, 61.4, 60.3, 57.2]
    J_C_1977_88_MAF = [CUMCS_MAC_conversion_factor*day_month_correction[index]*J_C_1977_88[index] for index in range(len(J_C_1977_88))]
    J_C_1977_88_MAF_Early_Kharif = J_C_1977_88_MAF[:7]
    J_C_1977_88_MAF_Late_Kharif = J_C_1977_88_MAF[7:]
    ##############################################
    
    def convert_to_MAF(data):
        conversion_factor = 43560   # Convert million cubic feet to MAF
        converted_data = {}
        for station, data_dict in data.items():
            converted_data[station] = {}
            for category, sub_dict in data_dict.items():  # Minimum, Maximum
                converted_data[station][category] = {}
                for season, arr in sub_dict.items():  # Early_Kharif, Late_Kharif
                    converted_data[station][category][season] = np.array(arr) / conversion_factor
        return converted_data

    Rim_prediction = convert_to_MAF({rm : IRSA_prediction__Kharif(rm, input_data, season, observed_flow_index, percentage_range) for rm in rim_station})
    J_C_Rim_stations_prediction = {key: Rim_prediction[key] for key in J_C_rim_stations_name}
    Indus_Rim_stations_prediction = {key: Rim_prediction[key] for key in Indus_rim_stations_name}
    Filling_withdraw_fraction_Mangla_Early_Kharif = Filling_withdraw_fraction_Mangla[:7]
    Filling_withdraw_fraction_Mangla_Late_Kharif = Filling_withdraw_fraction_Mangla[7:]
    
    Filling_withdraw_fraction_Tarbela_Early_Kharif = Filling_withdraw_fraction_Tarbela[:7]
    Filling_withdraw_fraction_Tarbela_Late_Kharif = Filling_withdraw_fraction_Tarbela[7:]
    if season == "Kharif":
        J_C_Shortage = {}
        Punjab_J_C_Canal_Wdls_Outflow_RQBS = {}
        Sindh_Channel_dis_df = {}
        Punjab_J_C_Channel_dis_df = {}
        Punjab_Indus_Channel_dis_df = {}
        scenarios = ["Maximum", "Minimum"]
        kharif_season_phases = ['Early_Kharif', 'Late_Kharif']



        for scenario in scenarios:
            System_Outflow_RQBS_min_max = {}
            Jhelum_at_Mangla_Early_Kharif = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario][kharif_season_phases[0]])
            Jhelum_at_Mangla_Late_Kharif = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario][kharif_season_phases[1]])
            
            Chenab_at_Marala_Early_Kharif = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario][kharif_season_phases[0]])
            Chenab_at_Marala_Late_Kharif = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario][kharif_season_phases[1]])
            
            Inflows_J_C_Command_Early_Kharif = Jhelum_at_Mangla_Early_Kharif + Chenab_at_Marala_Early_Kharif + sum(Eastern_rivers[0:7])
            Inflows_J_C_Command_Late_Kharif = Jhelum_at_Mangla_Late_Kharif + Chenab_at_Marala_Late_Kharif + sum(Eastern_rivers[7:])

            Storage_to_Fill_Mangla = Maximum_Storage_Mangla - Initial_Storage_Mangla 
            Storage_Fill_Mangla_Early_Kharif = Storage_to_fill_in_E_Kharif_Mangla * Storage_to_Fill_Mangla
            Storage_Fill_Mangla_Late_Kharif = Storage_to_fill_in_L_Kharif_Mangla * Storage_to_Fill_Mangla
            Storage_Dep_at_end_of_Season = 0.1 * Maximum_Storage_Mangla



            Storage_Release_Mangla = -1*(Storage_Fill_Mangla_Early_Kharif + Storage_Fill_Mangla_Late_Kharif) + Storage_Dep_at_end_of_Season
            
            System_Inflow_Early_Kharif = Inflows_J_C_Command_Early_Kharif - Storage_Fill_Mangla_Early_Kharif 
            System_Inflow_Late_Kharif = Inflows_J_C_Command_Late_Kharif - Storage_Fill_Mangla_Late_Kharif + Storage_Dep_at_end_of_Season
            System_Inflow = System_Inflow_Early_Kharif + System_Inflow_Late_Kharif
            Total_Availability_JC_Early_Kharif = System_Inflow_Early_Kharif * (1 - J_C_Early_Kharif_loss)
            Total_Availability_JC_Late_Kharif = System_Inflow_Late_Kharif * (1 - J_C_Late_Kharif_loss)
            Total_Availability_JC = Total_Availability_JC_Early_Kharif + Total_Availability_JC_Late_Kharif
            
            
            Indus_at_Tarbela_Early_Kharif = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario][kharif_season_phases[0]])
            Indus_at_Tarbela_Late_Kharif = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario][kharif_season_phases[1]])
            
            Kabul_at_Nowshehra_Early_Kharif = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario][kharif_season_phases[0]])
            Kabul_at_Nowshehra_Late_Kharif = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario][kharif_season_phases[1]])
             
            ##############################IIINNNN#################################################
            Total_Indus_Early_Kharif = Indus_at_Tarbela_Early_Kharif + Kabul_at_Nowshehra_Early_Kharif
            Total_Indus_Late_Kharif = Indus_at_Tarbela_Late_Kharif + Kabul_at_Nowshehra_Late_Kharif 
            
            #print(scenario, "********************", Indus_at_Tarbela_Early_Kharif, Kabul_at_Nowshehra_Early_Kharif)
            Storage_to_Fill_Tarbela = Maximum_Storage_Tarbela - Initial_Storage_Tarbela 
            Storage_Fill_Tarbela_Early_Kharif = Storage_to_fill_in_E_Kharif_Tarbela * Storage_to_Fill_Tarbela
            Storage_Fill_Tarbela_Late_Kharif = Storage_to_fill_in_L_Kharif_Tarbela * Storage_to_Fill_Tarbela

            Storage_Dep_at_end_of_Season_Tarbela_percent = 0.1
            Storage_Dep_at_end_of_Season_Tarbela = Storage_Dep_at_end_of_Season_Tarbela_percent*Maximum_Storage_Tarbela
            Storage_Release_Tarbela = -1*(Maximum_Storage_Tarbela*(1 - Storage_Dep_at_end_of_Season_Tarbela_percent) - Initial_Storage_Tarbela)
            ############################################################################################################### 
            JC_average_system_uses_1977_1982_Early_Kharif = JC_average_system_uses_1977_1982["Early_Kharif"]
            JC_average_system_uses_1977_1982_Late_Kharif = JC_average_system_uses_1977_1982["Late_Kharif"]
            
            Average_System_use_Indus_Early_Kharif = Average_System_use_Indus["Early_Kharif"]
            Average_System_use_Indus_Late_Kharif = Average_System_use_Indus["Late_Kharif"]
             
            Indus_Early_Kharif_loss = 1 - Indus_Early_Kharif_loss_percent
            Indus_Late_Kharif_loss = 1 - Indus_Late_Kharif_loss_percent
     
            Baloch_share_Early_Kharif = [day_month_correction[:7][index]*Baloch_share_historical[:7][index]*0.01983471 for index in range(len(KPK_share_historical[:7]))]
            KPK_share_Early_Kharif = [day_month_correction[:7][index]*KPK_share_historical[:7][index]*0.01983471 for index in range(len(KPK_share_historical[:7]))]
            
            Baloch_share_Late_Kharif = [day_month_correction[7:][index]*Baloch_share_historical[7:][index]*0.01983471 for index in range(len(KPK_share_historical[7:]))]
            KPK_share_Late_Kharif = [day_month_correction[7:][index]*KPK_share_historical[7:][index]*0.01983471 for index in range(len(KPK_share_historical[7:]))] 
            ###############################################################################################################  

            #################################Check shortage values################################################# 
            Indus_know_parameters_Early_Kharif = Indus_Early_Kharif_loss*(Storage_Release_Tarbela + Total_Indus_Early_Kharif) - KPK_Baloch_share_Early_Kharif - Below_Kotri_Early_Kharif
            Numerator_Early_Kharif = Total_Availability_JC_Early_Kharif*Average_System_use_Indus_Early_Kharif - JC_average_system_uses_1977_1982_Early_Kharif*Indus_know_parameters_Early_Kharif
            Denominator_Early_Kharif = Indus_Early_Kharif_loss*JC_average_system_uses_1977_1982_Early_Kharif + Average_System_use_Indus_Early_Kharif
            J_C_Outflow_Early_Kharif = Numerator_Early_Kharif/Denominator_Early_Kharif

        
            Indus_know_parameters_Late_Kharif = Indus_Late_Kharif_loss*(Storage_Release_Tarbela + Total_Indus_Late_Kharif) - KPK_Baloch_share_Late_Kharif - Below_Kotri_Late_Kharif
            Numerator_Late_Kharif = Total_Availability_JC_Late_Kharif*Average_System_use_Indus_Late_Kharif - JC_average_system_uses_1977_1982_Late_Kharif*Indus_know_parameters_Late_Kharif
            Denominator_Late_Kharif = Indus_Late_Kharif_loss*JC_average_system_uses_1977_1982_Late_Kharif + Average_System_use_Indus_Late_Kharif
            J_C_Outflow_Late_Kharif = Numerator_Late_Kharif/Denominator_Late_Kharif

            ###################################Check############################################################### 
            JC_Canal_Availability_Early_Kharif = Total_Availability_JC_Early_Kharif - J_C_Outflow_Early_Kharif
            #print(scenario, J_C_Outflow_Early_Kharif, "££££££££££££££££££££££££££££",Total_Availability_JC_Early_Kharif, JC_average_system_uses_1977_1982_Early_Kharif)
            J_C_Shortage_Early_Kharif = (1 - JC_Canal_Availability_Early_Kharif/JC_average_system_uses_1977_1982_Early_Kharif)/100
            
            JC_Canal_Availability_Late_Kharif = Total_Availability_JC_Late_Kharif - J_C_Outflow_Late_Kharif
            J_C_Shortage_Late_Kharif = (1 - JC_Canal_Availability_Late_Kharif/JC_average_system_uses_1977_1982_Late_Kharif)/100
            ###################################Check###############################################################  
            

            ############################Early_Khrif################################################
            #need to extract Initial_Storage_Mangla from the Mangla storage node@@@@@@@@@@@@@
            Mangla_Live_content_MAF_Early_Kharif = [Initial_Storage_Mangla]  #in MAC
            
            Initial_Storage_temp = Initial_Storage_Mangla
            
            for vol in Filling_withdraw_fraction_Mangla_Early_Kharif:
                release = (Storage_Fill_Mangla_Early_Kharif*vol/100)
                Initial_Storage_temp = release + Initial_Storage_temp 
                Mangla_Live_content_MAF_Early_Kharif.append(Initial_Storage_temp)

            Mangla_outflow_Early_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Mangla_Early_Kharif)):
                release = J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario][kharif_season_phases[0]][index] - (Mangla_Live_content_MAF_Early_Kharif[index+1] - Mangla_Live_content_MAF_Early_Kharif[index])
                Mangla_outflow_Early_Kharif.append(release)
            #need to combine the Early and Late kharif rim station predictions
            System_Inflow_Early_Kharif = [Mangla_outflow_Early_Kharif[index] + Eastern_rivers[0:7][index] + J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario][kharif_season_phases[0]][index] for index in range(len(Mangla_outflow_Early_Kharif))]
            Net_inflow_after_loss_Early_Kharif = [-1*inflow*(J_C_Early_Kharif_loss) for inflow in System_Inflow_Early_Kharif]
            Net_inflow_Early_Kharif = [System_Inflow_Early_Kharif[index] + Net_inflow_after_loss_Early_Kharif[index] for index in range(len(Net_inflow_after_loss_Early_Kharif))]
            
            
            ############################Early_Khrif################################################
            Punjab_J_C_Canal_Wdls_Early_Kharif = [
                Net_inflow_Early_Kharif[index] if interpolate_value(J_C_Shortage_Early_Kharif, index + 1) > Net_inflow_Early_Kharif[index] 
                else interpolate_value(J_C_Shortage_Early_Kharif, index + 1)
                for index in range(len(Net_inflow_Early_Kharif))
            ]
            System_Outflow_RQBS_Early_Kharif = [Net_inflow_Early_Kharif[index] - Punjab_J_C_Canal_Wdls_Early_Kharif[index] for index in range(len(Punjab_J_C_Canal_Wdls_Early_Kharif))]
            ############################Early_Khrif################################################

            #need to extract Initial_Storage_Mangla from the Mangla storage node@@@@@@@@@@@@@
            Mangla_Live_content_MAF_Late_Kharif = [Mangla_Live_content_MAF_Early_Kharif[-1]]  
            Initial_Storage_temp = Mangla_Live_content_MAF_Late_Kharif[0]

            for index in range(len(Filling_withdraw_fraction_Mangla_Late_Kharif)):
                if index > 7:
                    if index == 8:                    
                        Storage_Dep_at_end_of_season = Initial_Storage_Mangla * Storage_Dep_at_end_of_season_Mangla
                        Initial_Storage_temp = Mangla_Live_content_MAF_Early_Kharif[-1]
                        Mangla_Live_content_MAF_Late_Kharif.append(Initial_Storage_temp - Storage_Dep_at_end_of_season)
                    else:
                        Storage_Dep_at_end_of_season = Initial_Storage_Mangla*Storage_Dep_at_end_of_season_Mangla
                        Initial_Storage_temp = Mangla_Live_content_MAF_Late_Kharif[index-1]
                        Mangla_Live_content_MAF_Late_Kharif.append(Initial_Storage_temp - Storage_Dep_at_end_of_season)
                else:
                    vol = Filling_withdraw_fraction_Mangla_Late_Kharif[index]
                    release = (Storage_Fill_Mangla_Late_Kharif*vol/100)
                    Initial_Storage_temp = release + Initial_Storage_temp 
                    Mangla_Live_content_MAF_Late_Kharif.append(Initial_Storage_temp)

            Mangla_outflow_Late_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Mangla_Late_Kharif)):
                release = J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario][kharif_season_phases[1]][index] - (Mangla_Live_content_MAF_Late_Kharif[index+1] - Mangla_Live_content_MAF_Late_Kharif[index])
                Mangla_outflow_Late_Kharif.append(release)            
            #need to combine the Early and Late kharif rim station predictions
            
            System_Inflow_Late_Kharif = [Mangla_outflow_Late_Kharif[index] + Eastern_rivers[7:][index] + J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario][kharif_season_phases[1]][index] for index in range(len(Mangla_outflow_Late_Kharif))]
            Net_inflow_after_loss_Late_Kharif = [-1*inflow*(J_C_Late_Kharif_loss/100) for inflow in System_Inflow_Late_Kharif]
            Net_inflow_Late_Kharif = [System_Inflow_Late_Kharif[index] + Net_inflow_after_loss_Late_Kharif[index] for index in range(len(Net_inflow_after_loss_Late_Kharif))]

            Punjab_J_C_Canal_Wdls_Late_Kharif = []
            Punjab_J_C_Canal_Wdls_Late_Kharif = [
                Net_inflow_Late_Kharif[index] if (1 - J_C_Shortage_Late_Kharif * 0.01) * J_C_1977_88_MAF_Late_Kharif[index] > Net_inflow_Late_Kharif[index] 
                else (1 - J_C_Shortage_Late_Kharif * 0.01) * J_C_1977_88_MAF_Late_Kharif[index]
                for index in range(len(Net_inflow_Late_Kharif))
            ]
            System_Outflow_RQBS_Late_Kharif = [Net_inflow_Late_Kharif[index] - Punjab_J_C_Canal_Wdls_Late_Kharif[index] for index in range(len(Punjab_J_C_Canal_Wdls_Late_Kharif))]
            ############################Early_Khrif################################################

            ############################JJJJCCC################################################
            #JC_outflow is part of the below equation that need to be fixed
            
            Total_System_Inflows_Indus_Early_Kharif = Total_Indus_Early_Kharif + sum(System_Outflow_RQBS_Early_Kharif) - Storage_Fill_Tarbela_Early_Kharif
            Total_Availability_Indus_Early_Kharif = Total_System_Inflows_Indus_Early_Kharif - Indus_Early_Kharif_loss_percent*Total_System_Inflows_Indus_Early_Kharif
            
            
            #Implement paraII conditional statment
            Canal_Availability_Indus_Early_Kharif = Total_Availability_Indus_Early_Kharif - Below_Kotri_Early_Kharif

            Punjab_Sindh_share_Indus_Early_Kharif = Canal_Availability_Indus_Early_Kharif - KPK_Baloch_share_Early_Kharif
            Indus_Shortage_Early_Kharif = (1 - Punjab_Sindh_share_Indus_Early_Kharif/Average_System_use_Indus_Early_Kharif)*100

            Total_System_Inflows_Indus_Late_Kharif = Total_Indus_Late_Kharif + sum(System_Outflow_RQBS_Late_Kharif) - Storage_Fill_Tarbela_Late_Kharif
            Total_Availability_Indus_Late_Kharif = Total_System_Inflows_Indus_Late_Kharif - Indus_Late_Kharif_loss_percent*Total_System_Inflows_Indus_Late_Kharif
            Canal_Availability_Indus_Late_Kharif = Total_Availability_Indus_Late_Kharif - Below_Kotri_Late_Kharif
            Punjab_Sindh_share_Indus_Late_Kharif = Canal_Availability_Indus_Late_Kharif - KPK_Baloch_share_Late_Kharif
            Indus_Shortage_Late_Kharif = (1 - Punjab_Sindh_share_Indus_Late_Kharif/Average_System_use_Indus_Late_Kharif)*100

            ##############################IIINNNN#################################################
            ##############################IIINNNN##################################################
            #need to extract Initial_Storage_Tarbela from the Mangla storage node@@@@@@@@@@@@@
            
            Filling_withdraw_fraction_Tarbela_Early_Kharif_1 = Filling_withdraw_fraction_Tarbela[:7]
            Filling_withdraw_fraction_Tarbela_Early_Kharif_2 = Filling_withdraw_fraction_Tarbela[7:9]
            Filling_withdraw_fraction_Tarbela_Early_Kharif_3 = [4.529, 5.054, 5.602, Maximum_Storage_Tarbela, Maximum_Storage_Tarbela, Maximum_Storage_Tarbela]
            Filling_withdraw_fraction_Tarbela_Early_Kharif_4 = Filling_withdraw_fraction_Tarbela[15:18]


            Live_content_MAF_Early_Kharif = [Initial_Storage_Tarbela]  #in MAC
            Storage_Fill_Tarbela_Early_Kharif = 0.23*(Maximum_Storage_Tarbela - Initial_Storage_Tarbela) 
            Storage_Fill_Tarbela_Late_Kharif = 0.77*(Maximum_Storage_Tarbela - Initial_Storage_Tarbela) 

            for vol in Filling_withdraw_fraction_Tarbela_Early_Kharif_1:
                release = (Storage_Fill_Tarbela_Early_Kharif*vol/100)
                Initial_Storage_temp = release + Initial_Storage_temp 
                Live_content_MAF_Early_Kharif.append(Initial_Storage_temp)

            Live_content_MAF_Late_Kharif = [Live_content_MAF_Early_Kharif[-1]]
            Tarbela_filling_Limit_start = 4.025
            Live_content_MAF_constat = Live_content_MAF_Early_Kharif[-1]
            for vol in Filling_withdraw_fraction_Tarbela_Early_Kharif_2:
                release = ((Tarbela_filling_Limit_start - Live_content_MAF_constat)*vol/100) 
                Initial_Storage_temp = release + Initial_Storage_temp 
                Live_content_MAF_Late_Kharif.append(Initial_Storage_temp)


            for vol in Filling_withdraw_fraction_Tarbela_Early_Kharif_3:
                Live_content_MAF_Late_Kharif.append(vol)
            
            Storage_Dep_at_end_of_season = 0.1
            Tarbela_Max_Storage = 6.17
            Initial_Storage_temp = Live_content_MAF_Late_Kharif[-1]
            for vol in Filling_withdraw_fraction_Tarbela_Early_Kharif_4:
                release = (Initial_Storage_temp - Storage_Dep_at_end_of_season*Tarbela_Max_Storage*vol/100) 
                Initial_Storage_temp = release  
                Live_content_MAF_Late_Kharif.append(Initial_Storage_temp)


            Tarbela_outflow_Early_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Tarbela_Early_Kharif)):
                release = Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario][kharif_season_phases[0]][index]-(Live_content_MAF_Early_Kharif[index+1]-Live_content_MAF_Early_Kharif[index])
                Tarbela_outflow_Early_Kharif.append(release)

            Tarbela_outflow_Late_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Tarbela_Late_Kharif)):
                release = Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario][kharif_season_phases[1]][index]-(Live_content_MAF_Late_Kharif[index+1]-Live_content_MAF_Late_Kharif[index])
                Tarbela_outflow_Late_Kharif.append(release)

            ParaII = [68.2, 70.1, 79.1, 99.6, 116.5, 135.3, 162.8, 187.2, 198.7, 205.0, 186.2, 175.9, 169.6, 168.6, 175.7, 177.3, 175.1, 170.1]
            ParaII_Early_Kharif = ParaII[:7] 
            ParaII_Late_Kharif = ParaII[7:] 
            Initial_Storage_temp = Initial_Storage_Tarbela
            Chashma_storage_Early_Kharif = [2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            Chashma_storage_Late_Kharif = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, -2.0, 2.0, -5.0]

            Indus_Net_System_Inflow_Early_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Tarbela_Early_Kharif)):
                Inflow = (Tarbela_outflow_Early_Kharif[index] + Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario][kharif_season_phases[0]][index] + Chashma_storage_Early_Kharif[index] + System_Outflow_RQBS_Early_Kharif[index])*Indus_Early_Kharif_loss
                Indus_Net_System_Inflow_Early_Kharif.append(Inflow)

            Indus_Net_System_Inflow_Late_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Tarbela_Late_Kharif)):
                Inflow = (Tarbela_outflow_Late_Kharif[index] + Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario][kharif_season_phases[1]][index] + Chashma_storage_Late_Kharif[index] + System_Outflow_RQBS_Late_Kharif[index])*Indus_Late_Kharif_loss
                Indus_Net_System_Inflow_Late_Kharif.append(Inflow)
            
            Proposed_Canal_Wdls_Indus_Early_Kharif = [ParaII_Early_Kharif[index] if Indus_Net_System_Inflow_Early_Kharif[index]>ParaII_Early_Kharif[index] else Indus_Net_System_Inflow_Early_Kharif[index] for index in range(len(Indus_Net_System_Inflow_Early_Kharif))]     
            System_Outflow_DS_Kotri_Early_Kharif = [Indus_Net_System_Inflow_Early_Kharif[index] - Proposed_Canal_Wdls_Indus_Early_Kharif[index] for index in range(len(Indus_Net_System_Inflow_Early_Kharif))]
            Total_Share_Punjab_Sindh_Early_Kharif = [Proposed_Canal_Wdls_Indus_Early_Kharif[index] - KPK_share_Early_Kharif[index] - Baloch_share_Early_Kharif[index] for index in range(len(KPK_share_Early_Kharif))]

            Proposed_Canal_Wdls_Indus_Late_Kharif = [ParaII_Late_Kharif[index] if Indus_Net_System_Inflow_Late_Kharif[index]>ParaII_Late_Kharif[index] else Indus_Net_System_Inflow_Late_Kharif[index] for index in range(len(Indus_Net_System_Inflow_Late_Kharif))]     
            System_Outflow_DS_Kotri_Late_Kharif = [Indus_Net_System_Inflow_Late_Kharif[index] - Proposed_Canal_Wdls_Indus_Late_Kharif[index] for index in range(len(Indus_Net_System_Inflow_Late_Kharif))]
            Total_Share_Punjab_Sindh_Late_Kharif = [Proposed_Canal_Wdls_Indus_Late_Kharif[index] - KPK_share_Late_Kharif[index] - Baloch_share_Late_Kharif[index] for index in range(len(KPK_share_Late_Kharif))]
        
            Total_Share_Punjab_Sindh = Total_Share_Punjab_Sindh_Early_Kharif + Total_Share_Punjab_Sindh_Late_Kharif 

            #condition to implement the Para2 para14b and 
            Sindh_share_Indus_para_2_percent = [100 - Punjab_share_Indus_para_2_percent[index] for index in range(len(Punjab_share_Indus_para_2_percent))]
            Punjab_Indus_Canal_Wdls = [Total_Share_Punjab_Sindh[index]*Punjab_share_Indus_para_2_percent[index]*0.01 for index in range(len(Punjab_share_Indus_para_2_percent))]
            Sindh_share_Canal_Wdls  = [Total_Share_Punjab_Sindh[index]*Sindh_share_Indus_para_2_percent[index]*0.01 for index in range(len(Punjab_share_Indus_para_2_percent))]

            ##############################IIINNNN#######################################################################
            Punjab_J_C_Canal_Wdls = Punjab_J_C_Canal_Wdls_Early_Kharif + Punjab_J_C_Canal_Wdls_Late_Kharif 

            Sindh_Channel_dis_df[scenario] = input_data['Sindh_Channel_dis_plan_percentage'][18:].mul(Sindh_share_Canal_Wdls, axis=0)
            Punjab_J_C_Channel_dis_df[scenario] = input_data['J_C_Channel_dis_plan_percentage'][18:].mul(Punjab_J_C_Canal_Wdls, axis=0)
            Punjab_Indus_Channel_dis_df[scenario] = input_data['Indus_Channel_dis_plan_percentage'][18:].mul(Punjab_Indus_Canal_Wdls, axis=0)
            System_Outflow_RQBS_min_max[scenario] = System_Outflow_RQBS_Late_Kharif + System_Outflow_RQBS_Late_Kharif
                
        output = {}
        Sindh_Channel_dis_df_likely = (Sindh_Channel_dis_df[scenarios[0]] + Sindh_Channel_dis_df[scenarios[1]]) / 2
        Punjab_J_C_Channel_dis_df_likely = (Punjab_J_C_Channel_dis_df[scenarios[0]] + Punjab_J_C_Channel_dis_df[scenarios[1]]) / 2
        Punjab_Indus_Channel_dis_df_likely = (Punjab_Indus_Channel_dis_df[scenarios[0]] + Punjab_Indus_Channel_dis_df[scenarios[1]]) / 2
        RQBS_Canal_Outflow_likely =  [sum(x) / len(x) for x in zip(*System_Outflow_RQBS_min_max.values())]
        
        output["Sindh_Channel_dis_df_likely"] = Sindh_Channel_dis_df_likely
        output["Punjab_J_C_Channel_dis_df_likely"] = Punjab_J_C_Channel_dis_df_likely
        output["Punjab_Indus_Channel_dis_df_likely"] = Punjab_Indus_Channel_dis_df_likely
        output["RQBS_Canal_Outflow_likely"] = RQBS_Canal_Outflow_likely
  
    return output


class Kharif_water_allocation(Parameter):
    def __init__(self, model,KPK_Baloch_share_Early_Kharif,KPK_Baloch_share_Late_Kharif,Below_Kotri_Early_Kharif,
    Below_Kotri_Late_Kharif,Indus_Early_Kharif_loss_percent,Indus_Late_Kharif_loss_percent,
    J_C_Early_Kharif_loss, J_C_Late_Kharif_loss, Storage_to_fill_in_E_Kharif_Tarbela, Storage_to_fill_in_L_Kharif_Tarbela, Storage_to_fill_in_E_Kharif_Mangla, Storage_to_fill_in_L_Kharif_Mangla, 
    Punjab_channel_heads, Sindh_channel_heads, 
    input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, 
    Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, 
    percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, 
    Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, 
    KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, 
    System_losses_percent_Indus, System_losses_JC, **kwargs):
        
        super().__init__(model, **kwargs)


        self.KPK_Baloch_share_Early_Kharif = KPK_Baloch_share_Early_Kharif
        self.KPK_Baloch_share_Late_Kharif = KPK_Baloch_share_Late_Kharif
        self.Below_Kotri_Early_Kharif = Below_Kotri_Early_Kharif
        self.Below_Kotri_Late_Kharif = Below_Kotri_Late_Kharif
        self.Indus_Early_Kharif_loss_percent = Indus_Early_Kharif_loss_percent
        self.Indus_Late_Kharif_loss_percent = Indus_Late_Kharif_loss_percent


        self.J_C_Early_Kharif_loss = J_C_Early_Kharif_loss
        self.J_C_Late_Kharif_loss = J_C_Late_Kharif_loss
        self.input_data = input_data
        self.Mangla_reservoir = Mangla_reservoir
        self.Tarbela_reservoir = Tarbela_reservoir
        
        self.Punjab_channel_heads = Punjab_channel_heads
        self.Sindh_channel_heads = Sindh_channel_heads

        self.Storage_to_fill_in_E_Kharif_Mangla = Storage_to_fill_in_E_Kharif_Mangla
        self.Storage_to_fill_in_L_Kharif_Mangla = Storage_to_fill_in_L_Kharif_Mangla
        self.Storage_to_fill_in_E_Kharif_Tarbela = Storage_to_fill_in_E_Kharif_Tarbela
        self.Storage_to_fill_in_L_Kharif_Tarbela = Storage_to_fill_in_L_Kharif_Tarbela
        self.Sindh_channel_heads_Guddu = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Guddu"]}
        self.Sindh_channel_heads_Sukkur = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Sukkur"]}
        self.Sindh_channel_heads_Kotri = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Kotri"]}

        self.Sindh_channel_heads_node = {**self.Sindh_channel_heads_Guddu, **self.Sindh_channel_heads_Sukkur, **self.Sindh_channel_heads_Kotri}
        self.Punjab_channel_heads_node = {node_name:model._get_node_from_ref(model, node_name) for node_name in self.Punjab_channel_heads}
        

        self.Punjab_channel_heads_recorders_name = [node+"_rec" for node in self.Punjab_channel_heads]
        self.Sindh_channel_heads_recorders_name = [node+"_rec" for node in self.Sindh_channel_heads["Guddu"] + self.Sindh_channel_heads["Sukkur"] + self.Sindh_channel_heads["Kotri"]]
        self.Punjab_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Punjab_channel_heads_recorders_name}
        self.Sindh_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Sindh_channel_heads_recorders_name}
        

        self.Indus_at_Chashma = Indus_at_Chashma
        self.Storage_Dep_at_end_of_season_Mangla = Storage_Dep_at_end_of_season_Mangla
        self.Storage_Dep_at_end_of_Season_Tarbela = Storage_Dep_at_end_of_Season_Tarbela
        self.percentage_range = percentage_range
        self.Filling_withdraw_fraction_Tarbela = Filling_withdraw_fraction_Tarbela
        self.Filling_withdraw_fraction_Mangla = Filling_withdraw_fraction_Mangla
        
        self.Eastern_rivers = Eastern_rivers
        self.JC_average_system_uses_1977_1982 = JC_average_system_uses_1977_1982
        self.Average_System_use_Indus = Average_System_use_Indus
        self.KPK_Baloch_share = KPK_Baloch_share
        self.KPK_share_historical = KPK_share_historical
        self.Baloch_share_historical = Baloch_share_historical
        
        self.Below_Kotri = Below_Kotri
        self.Punjab_share_Indus_para_2_percent = Punjab_share_Indus_para_2_percent
        self.System_losses_percent_Indus = System_losses_percent_Indus
        self.System_losses_JC = System_losses_JC

    def setup(self):
        super().setup()

    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        ts = self.model.timestepper.current
        days_in_month = timestep.period.days_in_month
        start_date = str(ts.year)+"-"+str(ts.month)+"-"+str(ts.day)
        self.val = 0
        start_year = self.model.timestepper.start.year
        Initial_Storage_Mangla = self.Mangla_reservoir.volume[i]/43560
        Maximum_Storage_Mangla = self.Mangla_reservoir.max_volume/43560
        Initial_Storage_Tarbela = self.Tarbela_reservoir.volume[i]/43560
        Maximum_Storage_Tarbela = 314166/43560
        
        self.pubjab_abstructed_Rabi = {}
        self.sindh_abstructed_Rabi = {}
        
        self.pubjab_remaining_demand_Rabi = {}
        self.sindh_remaining_demand_Rabi = {}
        
        if ts.month == 9 and ts.day > 29:
            self.Rabi_season_start_date_index = ts.index
            
        if start_year < ts.year:
            if ts.month == 3 and ts.day > 30:
                self.Rabi_index = 0
                self.Punjab_Sindh_channel_heads_index = 0
                self.Rabi_season_start_date_index = ts.index

                water_balance_J_C_zone_output = Kharif_Indus_J_C_zone_water_balance(self.KPK_Baloch_share_Early_Kharif,
                    self.KPK_Baloch_share_Late_Kharif,
                    self.Below_Kotri_Early_Kharif,
                    self.Below_Kotri_Late_Kharif,
                    self.Indus_Early_Kharif_loss_percent,
                    self.Indus_Late_Kharif_loss_percent,
                    
                    self.J_C_Early_Kharif_loss, self.J_C_Late_Kharif_loss, self.Storage_to_fill_in_E_Kharif_Tarbela, self.Storage_to_fill_in_L_Kharif_Tarbela, self.Storage_to_fill_in_E_Kharif_Mangla, self.Storage_to_fill_in_L_Kharif_Mangla, self.input_data, Initial_Storage_Mangla, Maximum_Storage_Mangla, Initial_Storage_Tarbela, Maximum_Storage_Tarbela, self.Indus_at_Chashma, self.Storage_Dep_at_end_of_season_Mangla, self.Storage_Dep_at_end_of_Season_Tarbela, start_date, self.percentage_range, self.Filling_withdraw_fraction_Tarbela, self.Filling_withdraw_fraction_Mangla, self.Eastern_rivers, self.JC_average_system_uses_1977_1982, self.Average_System_use_Indus, self.KPK_Baloch_share, self.KPK_share_historical, self.Baloch_share_historical, self.Below_Kotri, self.Punjab_share_Indus_para_2_percent, self.System_losses_percent_Indus, self.System_losses_JC)
                
                
                self.Sindh_Channel_dis_df_likely = water_balance_J_C_zone_output["Sindh_Channel_dis_df_likely"] 
                self.Punjab_J_C_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_J_C_Channel_dis_df_likely"]
                self.Punjab_Indus_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_Indus_Channel_dis_df_likely"]
                self.RQBS_Canal_Outflow_likely = water_balance_J_C_zone_output["RQBS_Canal_Outflow_likely"]
                
                self.Sindh_total_allocated_water = self.Sindh_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_J_C_total_allocated_water = self.Punjab_J_C_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_Indus_total_allocated_water = self.Punjab_Indus_Channel_dis_df_likely.sum(axis=1).tolist()
            
            Rabi_start_time = pd.to_datetime(str(ts.year) + '-09-29')
            Rabi_end_time = pd.to_datetime(str(ts.year + 1) + '-03-30') 
            if Rabi_start_time <= ts.datetime <= Rabi_end_time:
               
                
                #water used sofar
                self.pubjab_abstructed_Rabi = {recorder:sum(self.Punjab_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Punjab_channel_heads_recorders}
                self.sindh_abstructed_Rabi = {recorder:sum(self.Sindh_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Sindh_channel_heads_recorders}
                
                #remining water that is required to satisfy the full demand
                self.pubjab_remaining_demand_Rabi = {node:sum(self.Punjab_channel_heads_node[node].max_flow.dataframe[ts.datetime : Rabi_end_time].values) for node in self.Punjab_channel_heads_node}
                self.sindh_remaining_demand_Rabi  = {node:sum(self.Sindh_channel_heads_node[node].max_flow.dataframe[ts.datetime : Rabi_end_time].values) for node in self.Sindh_channel_heads_node}
                
                
                #how much avaiable is avaiable to allocate according to IRSA's prediction 
                if ts.day == 10 or ts.day == 20:
                    self.Sindh_Channel_available_water_to_allocate = sum(self.Sindh_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_J_C_Channel_available_water_to_allocate = sum(self.Punjab_J_C_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_Indus_Channel_available_water_to_allocate = sum(self.Punjab_Indus_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
         
                    self.Rabi_index += 1
                    
                elif ts.day == days_in_month:
                    self.Sindh_Channel_available_water_to_allocate = sum(self.Sindh_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_J_C_Channel_available_water_to_allocate = sum(self.Punjab_J_C_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_Indus_Channel_available_water_to_allocate = sum(self.Punjab_Indus_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    
                    self.Rabi_index += 1
        
        #need the list of nodes in Punjab J-C and Indus Zone and Sindh Zone
        #check the next 183 days demand and if there less water reduce the demand by a certain fraction for those who are using more than the allocated amount
        #J-C outflow need to be tracked everytime and the value need to match with the allocated one
        
        #step1: collect all the recorders to get how much water is allocated 
        #step2: then create a number of 
        
        
        val = sum(self.pubjab_remaining_demand_Rabi.values())
        return val
            
    @classmethod
    def load(cls, model, data):
        Mangla_reservoir = model._get_node_from_ref(model, data.pop("Mangla_reservoir_node"))
        Tarbela_reservoir = model._get_node_from_ref(model, data.pop("Tarbela_reservoir_node"))
        
        Indus_at_Chashma = data.pop("Indus_at_Chashma")
        Storage_Dep_at_end_of_season_Mangla = data.pop("Storage_Dep_at_end_of_season_Mangla")
        Storage_Dep_at_end_of_Season_Tarbela = data.pop("Storage_Dep_at_end_of_Season_Tarbela")
        System_losses = data.pop("System_losses")
        
        percentage_range = data.pop("percentage_range")
        Filling_withdraw_fraction_Tarbela = data.pop("Filling_withdraw_fraction_Tarbela")
        Filling_withdraw_fraction_Mangla = data.pop("Filling_withdraw_fraction_Mangla")
        Eastern_rivers = data.pop("Eastern_rivers")
        
        JC_average_system_uses_1977_1982 = data.pop("JC_average_system_uses_1977_1982")
        Average_System_use_Indus = data.pop("Average_System_use_Indus")
        KPK_Baloch_share = data.pop("KPK_Baloch_share")
        KPK_share_historical = data.pop("KPK_share_historical")
        
        Baloch_share_historical = data.pop("Baloch_share_historical")
        Below_Kotri = data.pop("Below_Kotri")
        Punjab_share_Indus_para_2_percent = data.pop("Punjab_share_Indus_para_2_percent")
        System_losses_percent_Indus = data.pop("System_losses_percent_Indus")
        System_losses_JC = data.pop("System_losses_JC")

        Storage_to_fill_in_E_Kharif_Tarbela = data.pop("Storage_to_fill_in_E_Kharif_Tarbela")
        Storage_to_fill_in_L_Kharif_Tarbela = data.pop("Storage_to_fill_in_L_Kharif_Tarbela")

        Storage_to_fill_in_E_Kharif_Mangla = data.pop("Storage_to_fill_in_E_Kharif_Mangla")
        Storage_to_fill_in_L_Kharif_Mangla = data.pop("Storage_to_fill_in_L_Kharif_Mangla")
        input_data = data.pop("url")
        Punjab_channel_heads = data.pop("Punjab_channel_heads")
        Sindh_channel_heads = data.pop("Sindh_channel_heads")

        J_C_Early_Kharif_loss = data.pop("J_C_Early_Kharif_loss")
        J_C_Late_Kharif_loss = data.pop("J_C_Late_Kharif_loss")

        KPK_Baloch_share_Early_Kharif = data.pop("KPK_Baloch_share_Early_Kharif")
        KPK_Baloch_share_Late_Kharif = data.pop("KPK_Baloch_share_Late_Kharif")
        Below_Kotri_Early_Kharif = data.pop("Below_Kotri_Early_Kharif")
        Below_Kotri_Late_Kharif = data.pop("Below_Kotri_Late_Kharif")
        Indus_Early_Kharif_loss_percent = data.pop("Indus_Early_Kharif_loss_percent")
        Indus_Late_Kharif_loss_percent = data.pop("Indus_Late_Kharif_loss_percent")
        return cls(model, KPK_Baloch_share_Early_Kharif, KPK_Baloch_share_Late_Kharif, Below_Kotri_Early_Kharif, Below_Kotri_Late_Kharif, Indus_Early_Kharif_loss_percent, Indus_Late_Kharif_loss_percent, J_C_Early_Kharif_loss, J_C_Late_Kharif_loss, Storage_to_fill_in_E_Kharif_Tarbela, Storage_to_fill_in_L_Kharif_Tarbela, Storage_to_fill_in_E_Kharif_Mangla, Storage_to_fill_in_L_Kharif_Mangla, Punjab_channel_heads, Sindh_channel_heads, input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, System_losses_percent_Indus, System_losses_JC, **data) 
Kharif_water_allocation.register()

class CSVRecorder(Recorder):
    """
    A Recorder that saves Node values to a CSV file.

    This class uses the csv package from the Python standard library

    Parameters
    ----------

    model : `pywr.model.Model`
        The model to record nodes from.
    csvfile : str
        The path to the CSV file.
    scenario_index : int
        The scenario index of the model to save.
    nodes : iterable (default=None)
        An iterable of nodes to save data. It defaults to None which is all nodes in the model
    kwargs : Additional keyword arguments to pass to the `csv.writer` object

    """
    def __init__(self, model, csvfile, scenario_index=0, nodes=None, complib=None, complevel=9, **kwargs):
        super(CSVRecorder, self).__init__(model, **kwargs)
        self.csvfile = csvfile
        self.scenario_index = scenario_index
        self.nodes = nodes
        self.csv_kwargs = kwargs.pop('csv_kwargs', {})
        self._node_names = None
        self._fh = None
        self._writer = None
        self.complib = complib
        self.complevel = complevel

    @classmethod
    def load(cls, model, data):
        url = data.pop("url")
        if not os.path.isabs(url) and model.path is not None:
            url = os.path.join(model.path, url)
        return cls(model, url, **data)

    def setup(self):
        """
        Setup the CSV file recorder.
        """

        if self.nodes is None:
            self._node_names = sorted(self.model.nodes.keys())
        else:
            node_names = []
            for node_ in self.nodes:
                # test if the node name is provided
                if isinstance(node_, str):
                    # lookup node by name
                    node_names.append(node_)
                else:
                    node_names.append((node_[1].name))
            self._node_names = node_names

    def reset(self):
        kwargs = {"newline": "", "encoding": "utf-8"}
        mode = "wt"

        if self.complib == "gzip":
            self._fh = gzip.open(self.csvfile, mode, self.complevel, **kwargs)
        elif self.complib in ("bz2", "bzip2"):
            self._fh = bz2.open(self.csvfile, mode, self.complevel, **kwargs)
        elif self.complib is None:
            self._fh = open(self.csvfile, mode, **kwargs)
        else:
            raise KeyError("Unexpected compression library: {}".format(self.complib))
        self._writer = csv.writer(self._fh, **self.csv_kwargs)
        # Write header data
        row = ["Datetime"] + [name for name in self._node_names]
        self._writer.writerow(row)

    def after(self):
        """
        Write the node values to the CSV file
        """
        values = [self.model.timestepper.current.datetime.isoformat()]
        for node_name in self._node_names:
            node = self.model.nodes[node_name]
            if isinstance(node, AbstractStorage):
                values.append(node.volume[self.scenario_index])
            elif isinstance(node, AbstractNode):
                
                values.append(node.flow[self.scenario_index-1])
            else:
                raise ValueError("Unrecognised Node type '{}' for CSV writer".format(type(node)))

        self._writer.writerow(values)

    def finish(self):
        if self._fh:
            self._fh.close()
CSVRecorder.register()

class HydropowerTargetParameterIndus(Parameter):
    """ A parameter that returns flow from a hydropower generation target.

    Same as HydropowerTargetParameter, except that the head unit is converted from feet to meters.

    """
    def __init__(self, model, target, water_elevation_parameter=None, max_flow=None, min_flow=None,
                 turbine_elevation=0.0, efficiency=1.0, density=1000, min_head=0.0,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(HydropowerTargetParameterIndus, self).__init__(model, **kwargs)

        self.target = target
        self.water_elevation_parameter = water_elevation_parameter
        self.max_flow = max_flow
        self.min_flow = min_flow
        self.min_head = min_head
        self.turbine_elevation = turbine_elevation
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion

    def value(self, ts, scenario_index):
        power = self.target.get_value(scenario_index)

        if self.water_elevation_parameter is not None:
            head = self.water_elevation_parameter.get_value(scenario_index)
            if self.turbine_elevation is not None:
                head -= self.turbine_elevation
        elif self.turbine_elevation is not None:
            head = self.turbine_elevation
        else:
            raise ValueError('One or both of storage_node or level must be set.')

        # -ve head is not valid
        head = max(head, 0.0)

        head = head * 0.3048 # feet to metre

        # Apply minimum head threshold.
        if head < self.min_head:
            return 0.0

        # Get the flow from the current node
        q = inverse_hydropower_calculation(power, head, 0.0, self.efficiency, density=self.density,
                                           flow_unit_conversion=self.flow_unit_conversion,
                                           energy_unit_conversion=self.energy_unit_conversion)

        # Bound the flow if required
        if self.max_flow is not None:
            q = min(self.max_flow.get_value(scenario_index), q)
        if self.min_flow is not None:
            q = max(self.min_flow.get_value(scenario_index), q)

        assert q >= 0.0

        return q

    @classmethod
    def load(cls, model, data):

        target = load_parameter(model, data.pop("target"))
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        if "max_flow" in data:
            max_flow = load_parameter(model, data.pop("max_flow"))
        else:
            max_flow = None

        if "min_flow" in data:
            min_flow = load_parameter(model, data.pop("min_flow"))
        else:
            min_flow = None

        return cls(model, target, water_elevation_parameter=water_elevation_parameter,
                   max_flow=max_flow, min_flow=min_flow, **data)
HydropowerTargetParameterIndus.register()

class HydropowerRecorderIndus(NumpyArrayNodeRecorder):
    """ Calculates the power production using the hydropower equation

    Same as HydropowerRecorder, except that the head unit is converted from feet to meters.

    """
    
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(HydropowerRecorderIndus, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
        self.turbine_elevation = turbine_elevation
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion
        
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        
    def setup(self):

        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        self._data = np.zeros((nts, ncomb))
        
    def reset(self):

        self._data[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        flow = self.node.flow

        for scenario_index in self.model.scenarios.combinations:

            if self.water_elevation_parameter is not None:
                head = self.water_elevation_parameter.get_value(scenario_index)
                if self.turbine_elevation is not None:
                    head -= self.turbine_elevation
            elif self.turbine_elevation is not None:
                head = self.turbine_elevation
            else:
                raise ValueError('One or both of storage_node or level must be set.')

            # -ve head is not valid
            head = max(head, 0.0)

            head = head * 0.3048 # feet to metre

            # Get the flow from the current node
            q = self.node.flow[scenario_index.global_id]
            power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                             flow_unit_conversion=self.flow_unit_conversion,
                                             energy_unit_conversion=self.energy_unit_conversion)

            self._data[ts.index, scenario_index.global_id] = power
            
        return 0
        
    def values(self):
        """Compute a value for each scenario using `temporal_agg_func`.
        """
        return self._temporal_aggregator.aggregate_2d(self._data, axis=0, ignore_nan=self.ignore_nan)
        
    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter, **data)
HydropowerRecorderIndus.register()

class TotalHydroEnergyRecorderIndus(BaseConstantNodeRecorder):
    """ Calculates the total energy production using the hydropower equation from a model run.

    Same as TotalHydroEnergyRecorder, except that the head unit is converted from feet to meters.

    """
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(TotalHydroEnergyRecorderIndus, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
        self.turbine_elevation = turbine_elevation
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion
        
    def setup(self):
        self._values = np.zeros(len(self.model.scenarios.combinations))
        
    def reset(self):
        self._values[...] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        days = ts.days
        flow = self.node.flow

        for scenario_index in self.model.scenarios.combinations:

            if self.water_elevation_parameter is not None:
                head = self.water_elevation_parameter.get_value(scenario_index)
                if self.turbine_elevation is not None:
                    head -= self.turbine_elevation
            elif self.turbine_elevation is not None:
                head = self.turbine_elevation
            else:
                raise ValueError('One or both of storage_node or level must be set.')

            # -ve head is not valid
            head = max(head, 0.0)

            head = head * 0.3048 # feet to metre

            # Get the flow from the current node
            q = self.node.flow[scenario_index.global_id]
            power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                             flow_unit_conversion=self.flow_unit_conversion,
                                             energy_unit_conversion=self.energy_unit_conversion)

            self._values[scenario_index.global_id] += power * days * 24
            
        return 0
            
    def values(self):
        return self._values

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter, **data)
TotalHydroEnergyRecorderIndus.register()

class renewable_dataframe(Parameter):
 
    def __init__(self, model, url, key, **kwargs):
        super().__init__(model, **kwargs)
        self.profile = url
        self.key = key
       
    def setup(self):
        super().setup()
        self.profile = pd.read_hdf(self.profile, key=self.key)
        self.hour = 0
 
    def value(self, ts, scenario_index):
       
        ts = self.model.timestepper.current
        month = ts.month
        day = ts.day
       
        if self.hour == 24:
            self.hour = 0
           
        hour = self.hour
       
        index = str(month) + '-' + str(day) + '-' + str(hour)
 
        # Make sure index exists
        if index not in self.profile.index:
            raise KeyError(f"Index {index} not found in profile HDF5")
 
        #print(self.profile)
       
        profile = self.profile.loc[index]
       
        self.hour += 1
       
        return profile
 
    @classmethod
    def load(cls, model, data):
 
        url = data.pop("url")
        key = data.pop("key")
 
        return cls(model, url, key, **data)
renewable_dataframe.register()

"""
=========================================================================
Calibration recorders
=========================================================================
"""

class MetricRecorderMixin:
    """
    Mixin providing common functionality for "comparison recorders" that compute
    a scalar (or per-scenario) performance metric between:

      - an observed time series loaded from file; and
      - a simulated time series stored by a `NumpyArray*Recorder`.

    Design notes
    ------------
    - This mixin centralises time alignment and merging logic so metric-specific
      recorders only implement `calculate_metric(obs, sim)`.
    - The simulated data are taken from `self.data` (recorded during the run).
    - The observed data are provided at construction time via the `observed`
      keyword (loaded by `load`).
    - Alignment is prepared during `setup()` and performed during `finish()`,
      enabling `values()` to remain numeric and lightweight.

    Units requirement (must be checked)
    -----------------------------------
    Observed values MUST be in the same physical units as the simulated values
    stored by this recorder (e.g., both in m³/s, or both in Mm³ per month).
    This implementation enforces a runtime scale sanity-check by comparing the
    median absolute magnitude of the aligned observed and simulated series.

    If the magnitude ratio is outside `units_check_ratio_bounds`, a ValueError is
    raised, because a unit mismatch (or inconsistent aggregation such as mean vs sum)
    is the most common cause of invalid metrics.

    You can disable or adjust this check via:
        - units_check=False
        - units_check_ratio_bounds=(lower, upper)

    Parameters
    ----------
    observed : pandas.Series
        Observed values indexed by datetime-like values (or values convertible
        to datetime). The index should represent the comparison timestep scale
        unless you deliberately choose to only resample the model to match.
    obs_freq : str or None
        Optional pandas offset alias used to resample model outputs prior to
        comparison (e.g., "M", "MS", "D"). If None, no resampling is performed.
    units_check : bool
        If True, enforce a scale sanity-check after alignment is built in `finish()`.
    units_check_ratio_bounds : tuple[float, float]
        (lower, upper) bounds for acceptable ratio of observed/simulated magnitude.
        Default is (1e-3, 1e3). Ratios outside this range raise a ValueError.

    Expected shapes
    ---------------
    - Observed aligned array: (T,)
    - Simulated aligned array: (T, S) where S is the number of scenarios

    Missing values policy
    ---------------------
    Metrics are computed using pairwise deletion per scenario:
      - For scenario s, only timesteps with finite obs AND finite sim[:, s]
        contribute to that scenario’s metric.
      - If there are insufficient valid points, that scenario’s metric is NaN.

    Subclass contract
    -----------------
    Subclasses must implement:
        calculate_metric(self, obs: np.ndarray, sim: np.ndarray) -> np.ndarray|float
    """

    def __init__(
        self,
        *args,
        observed=None,
        obs_freq=None,
        units_check=True,
        units_check_ratio_bounds=(1e-3, 1e3),
        **kwargs,
    ):
        """
        Store observed time series and comparison configuration.

        Notes
        -----
        This mixin does not call `super().__init__()` because the concrete base
        recorders (`NumpyArrayNodeRecorder`, `NumpyArrayStorageRecorder`) are
        initialised explicitly in the base comparison classes.
        """
        self.observed_original = observed
        self.obs_freq = obs_freq

        self.units_check = bool(units_check)
        self.units_check_ratio_bounds = tuple(units_check_ratio_bounds)

        # Prepared in setup()
        self._obs_df_prepared = None
        self._model_native_index = None

        # Cached alignment (built in finish())
        self._cached_obs_aligned = None
        self._cached_sim_aligned = None
        self._alignment_cache_ready = False

        # Cached metric values (computed on first values() call after finish())
        self._cached_metric_values = None

    def reset(self):
        """
        Reset recorder state for a new run.

        This clears any cached alignment and metric values, and then delegates
        to the parent recorder's `reset()` method.
        """
        self._cached_obs_aligned = None
        self._cached_sim_aligned = None
        self._alignment_cache_ready = False
        self._cached_metric_values = None

        super().reset()

    def setup(self):
        """
        Recorder setup hook.

        Improvements implemented
        ------------------------
        - Observed preprocessing is performed here (index parsing/coercion),
          aligning with Pywr's lifecycle expectations.
        - Model datetime index is also prepared here.

        The final merge/alignment is built in `finish()` once model data are complete.
        """
        super().setup()

        # Prepare observed once (do not mutate original).
        obs_df = self.observed_original.copy().to_frame()

        # Ensure datetime-like index if possible.
        if not pd.api.types.is_datetime64_any_dtype(obs_df.index):
            obs_df.index = pd.to_datetime(obs_df.index)

        # Standardise dtype to numeric (coerce errors to NaN).
        obs_df.iloc[:, 0] = pd.to_numeric(obs_df.iloc[:, 0], errors="coerce")

        self._obs_df_prepared = obs_df

        # Prepare model native datetime index once.
        native_index = self.model.timestepper.datetime_index
        if isinstance(native_index, pd.PeriodIndex):
            native_index = native_index.to_timestamp()
        else:
            # Ensure datetime64 dtype where possible.
            if not pd.api.types.is_datetime64_any_dtype(native_index):
                native_index = pd.to_datetime(native_index)

        self._model_native_index = native_index

    def finish(self):
        """
        Recorder finish hook.

        Improvements implemented
        ------------------------
        - Alignment/merge is performed once per run (after `self.data` is populated).
        - Alignment results are cached for subsequent `values()` calls.
        - A unit-consistency (scale) check is enforced here.

        Notes
        -----
        This method is called after the model run completes, which is the earliest
        point at which `self.data` contains the full simulated time series.
        """
        super().finish()

        self._build_alignment_cache()
        self._enforce_units_check()

    def _build_alignment_cache(self):
        """
        Build and cache aligned observed and simulated arrays.

        Alignment strategy (as implemented)
        ----------------------------------
        1. Use the preprocessed observed DataFrame from `setup()`.
        2. Construct a model DataFrame from `self.data` using the model's
           native datetime index.
        3. If `obs_freq` is provided, resample the model DataFrame using mean.
           If the model timestep frequency differs from `obs_freq`, convert
           indices to "YYYY-MM" strings to force month-based alignment.
        4. Merge observed and model DataFrames on index (inner join).
        5. Drop rows where observed is not finite (never usable for metrics).

        Cached outputs
        --------------
        self._cached_obs_aligned : np.ndarray | None
            1D array of aligned observed values (T,).
        self._cached_sim_aligned : np.ndarray | None
            2D array of aligned simulated values (T, S).
        """
        freq = self.obs_freq

        obs = self._obs_df_prepared
        if obs is None:
            # setup() not called or failed
            self._cached_obs_aligned = None
            self._cached_sim_aligned = None
            self._alignment_cache_ready = True
            return

        # Build model DataFrame from recorded data.
        mod_df = pd.DataFrame(self.data, index=self._model_native_index)

        # Apply resampling / index coercion logic.
        if freq is None:
            # No resampling requested
            obs_keyed = obs
            mod_keyed = mod_df
        else:
            if self.model.timestepper.freq == freq:
                mod_df = mod_df.resample(freq).mean()
                obs_keyed = obs
                mod_keyed = mod_df
            else:
                # Resample model to requested frequency then key both sides by YYYY-MM.
                mod_df.index = mod_df.index.astype("datetime64[ns]")
                mod_df = mod_df.resample(freq).mean()
                mod_df.index = mod_df.index.strftime("%Y-%m")

                obs_keyed = obs.copy()
                obs_keyed.index = obs_keyed.index.astype("datetime64[ns]").strftime("%Y-%m")

                mod_keyed = mod_df

        merged = pd.merge(obs_keyed, mod_keyed, how="inner", left_index=True, right_index=True)

        if merged.empty:
            self._cached_obs_aligned = None
            self._cached_sim_aligned = None
            self._alignment_cache_ready = True
            return

        obs_aligned = merged.iloc[:, 0].to_numpy()
        sim_aligned = merged.iloc[:, 1:].to_numpy()

        # Improvement 3 (partial): remove rows where observed is NaN/inf.
        # (Scenario-specific pairwise deletion is handled in metric calculations.)
        mask_obs = np.isfinite(obs_aligned)
        obs_aligned = obs_aligned[mask_obs]
        sim_aligned = sim_aligned[mask_obs, :]

        if obs_aligned.size == 0:
            self._cached_obs_aligned = None
            self._cached_sim_aligned = None
        else:
            self._cached_obs_aligned = obs_aligned
            self._cached_sim_aligned = sim_aligned

        self._alignment_cache_ready = True

    def _enforce_units_check(self):
        """
        Enforce a basic unit/scale sanity check.

        Rationale
        ---------
        The metrics assume observed and simulated data are directly comparable.
        If the observed series is in different units (or different aggregation,
        e.g., monthly total vs monthly mean), metrics are invalid.

        Implementation
        --------------
        Compare the median absolute magnitude of observed and simulated aligned data.
        If ratio outside `units_check_ratio_bounds`, raise ValueError.

        Notes
        -----
        This is a sanity check, not a physical unit conversion mechanism.
        """
        if not self.units_check:
            return

        if not self._alignment_cache_ready:
            return

        obs = self._cached_obs_aligned
        sim = self._cached_sim_aligned

        if obs is None or sim is None:
            return

        lower, upper = self.units_check_ratio_bounds

        # Robust scale estimators
        scale_obs = np.nanmedian(np.abs(obs))

        # For simulated, take median across timesteps and scenarios then median across scenarios.
        scale_sim_per_scenario = np.nanmedian(np.abs(sim), axis=0)
        scale_sim = np.nanmedian(scale_sim_per_scenario)

        # If either series is (effectively) all zeros, scale check is not informative.
        if not np.isfinite(scale_obs) or not np.isfinite(scale_sim) or scale_obs == 0 or scale_sim == 0:
            return

        ratio = scale_obs / scale_sim

        if ratio < lower or ratio > upper:
            raise ValueError(
                "Observed and simulated series appear to be on different scales. "
                "This usually indicates a unit mismatch (e.g., m³/s vs Mm³/month) or "
                "inconsistent aggregation (e.g., monthly mean vs monthly sum). "
                f"Scale ratio (obs/sim)={ratio:.3g} is outside bounds {self.units_check_ratio_bounds}. "
                "Verify units and aggregation are consistent, or adjust/disable the check via "
                "`units_check=False` or `units_check_ratio_bounds=(lower, upper)`."
            )

    def _get_merged_data(self):
        """
        Return cached aligned observed and simulated arrays.

        Improvements implemented
        ------------------------
        - Alignment is built once in `finish()` and cached.
        - If cache is not ready (e.g., finish not called), build it on demand.

        Returns
        -------
        obs_aligned : np.ndarray or None
            1D array of aligned observed values (T,).
        sim_aligned : np.ndarray or None
            2D array of aligned simulated values (T, S).
        """
        if self._alignment_cache_ready:
            return self._cached_obs_aligned, self._cached_sim_aligned

        # On-demand build (defensive), then return.
        self._build_alignment_cache()
        return self._cached_obs_aligned, self._cached_sim_aligned

    def values(self):
        """
        Compute the metric for this recorder.

        Returns
        -------
        np.ndarray
            A 1D numpy array of metric values per scenario, shape (S,).
            If the metric calculation returns a scalar, it is wrapped as (1,).

        Improvements implemented
        ------------------------
        - Result caching: subsequent calls return cached metric values without
          recomputing alignment or metrics.
        - Numeric-only: alignment is expected to be precomputed in `finish()`.

        Notes
        -----
        - If alignment yields no overlapping timestamps, returns [np.nan].
        - Subclasses must implement `calculate_metric(obs, sim)`.
        """
        if self._cached_metric_values is not None:
            return self._cached_metric_values

        obs, sim = self._get_merged_data()

        if obs is None or sim is None:
            self._cached_metric_values = np.array([np.nan])
            return self._cached_metric_values

        val = self.calculate_metric(obs, sim)

        if np.isscalar(val):
            self._cached_metric_values = np.array([val])
        else:
            self._cached_metric_values = np.array(val)

        return self._cached_metric_values

    # def aggregated_value(self):
    #     """
    #     Return a single scalar metric value.

    #     Behaviour
    #     ---------
    #     - If there is a single scenario value, return it as a float.
    #     - If multiple scenarios are present, return the mean across scenarios.

    #     Returns
    #     -------
    #     float
    #         Scalar aggregated metric value.
    #     """
    #     val_array = self.values()

    #     if val_array.size == 1:
    #         return float(val_array[0])

    #     return float(np.mean(val_array))

    @classmethod
    def load(cls, model, data):
        """
        Pywr JSON loader for comparison recorders.

        Expected JSON fields (in `data`)
        --------------------------------
        observed : str
            Column name in the source file containing observed values.
        index_col : str
            Column name in the source file containing the datetime index.
        url : str
            Path/URL to the observed data file (.csv, .xlsx, .xls).
        obs_freq : str, optional
            Resampling frequency (pandas offset alias).
        node : str|dict
            Node reference resolvable by `model._get_node_from_ref`.

        Optional JSON fields (in `data`)
        --------------------------------
        units_check : bool
            Enable/disable the enforced unit/scale sanity check.
        units_check_ratio_bounds : list[float, float] or tuple[float, float]
            Acceptable bounds for observed/simulated magnitude ratio.

        Returns
        -------
        cls
            Instantiated recorder object.

        Important
        ---------
        Observed values MUST be in the same units as the simulated values recorded
        by this recorder. This implementation does not perform unit conversion.
        """
        observed_key = data.pop("observed")
        index_col = data.pop("index_col")
        url = data.pop("url")
        obs_freq = data.pop("obs_freq", None)

        # Optional enforcement configuration
        units_check = data.pop("units_check", True)
        units_check_ratio_bounds = data.pop("units_check_ratio_bounds", (1e-3, 1e3))

        if ".csv" in url:
            df_raw = pd.read_csv(url)
        elif ".xlsx" in url or ".xls" in url:
            df_raw = pd.read_excel(url)
        else:
            raise ValueError(f"Unsupported file format: {url}")

        if index_col not in df_raw.columns:
            raise KeyError(f"Index column '{index_col}' not found in file.")

        df_raw[index_col] = pd.to_datetime(df_raw[index_col])
        df_raw.set_index(index_col, inplace=True)

        observed_series = pd.to_numeric(df_raw[observed_key], errors="coerce")
        node = model._get_node_from_ref(model, data.pop("node"))

        # Ensure bounds is a tuple (JSON likely provides list)
        if isinstance(units_check_ratio_bounds, list):
            units_check_ratio_bounds = tuple(units_check_ratio_bounds)

        return cls(
            model,
            node,
            observed=observed_series,
            obs_freq=obs_freq,
            units_check=units_check,
            units_check_ratio_bounds=units_check_ratio_bounds,
            **data,
        )

class BaseComparisonNodeRecorder(MetricRecorderMixin, NumpyArrayNodeRecorder):
    """
    Base comparison recorder for Node flows.

    Inherits storage of simulated node timeseries from `NumpyArrayNodeRecorder`
    and metric computation/alignment behaviour from `MetricRecorderMixin`.
    """

    def __init__(
        self,
        model,
        node,
        observed,
        obs_freq=None,
        units_check=True,
        units_check_ratio_bounds=(1e-3, 1e3),
        **kwargs,
    ):
        """Initialise with observed series and optional comparison configuration."""
        MetricRecorderMixin.__init__(
            self,
            observed=observed,
            obs_freq=obs_freq,
            units_check=units_check,
            units_check_ratio_bounds=units_check_ratio_bounds,
        )
        NumpyArrayNodeRecorder.__init__(self, model, node, **kwargs)


class BaseComparisonStorageRecorder(MetricRecorderMixin, NumpyArrayStorageRecorder):
    """
    Base comparison recorder for Storage node volumes.

    Inherits storage of simulated storage timeseries from `NumpyArrayStorageRecorder`
    and metric computation/alignment behaviour from `MetricRecorderMixin`.
    """

    def __init__(
        self,
        model,
        node,
        observed,
        obs_freq=None,
        units_check=True,
        units_check_ratio_bounds=(1e-3, 1e3),
        **kwargs,
    ):
        """Initialise with observed series and optional comparison configuration."""
        MetricRecorderMixin.__init__(
            self,
            observed=observed,
            obs_freq=obs_freq,
            units_check=units_check,
            units_check_ratio_bounds=units_check_ratio_bounds,
        )
        NumpyArrayStorageRecorder.__init__(self, model, node, **kwargs)


class RootMeanSquaredErrorNodeRecorder(BaseComparisonNodeRecorder):
    """
    RMSE metric for Node flows.

    RMSE = sqrt( mean( (obs - sim)^2 ) )

    Returns per-scenario RMSE values.

    Missing values
    --------------
    Pairwise deletion is applied implicitly: any timestep with NaN in either
    obs or sim for scenario s is ignored for that scenario.
    """

    def calculate_metric(self, obs, sim):
        # obs: (T,), sim: (T, S)
        err2 = (obs[:, np.newaxis] - sim) ** 2
        return np.sqrt(np.nanmean(err2, axis=0))


RootMeanSquaredErrorNodeRecorder.register()


class RootMeanSquaredErrorStorageRecorder(BaseComparisonStorageRecorder):
    """
    RMSE metric for Storage node volumes.

    RMSE = sqrt( mean( (obs - sim)^2 ) )

    Returns per-scenario RMSE values.

    Missing values
    --------------
    Pairwise deletion is applied implicitly: any timestep with NaN in either
    obs or sim for scenario s is ignored for that scenario.
    """

    def calculate_metric(self, obs, sim):
        err2 = (obs[:, np.newaxis] - sim) ** 2
        return np.sqrt(np.nanmean(err2, axis=0))


RootMeanSquaredErrorStorageRecorder.register()


class NashSutcliffeEfficiencyNodeRecorder(BaseComparisonNodeRecorder):
    """
    Nash–Sutcliffe Efficiency (NSE) metric for Node flows.

    NSE = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)

    Returns per-scenario NSE values.

    Missing values
    --------------
    Pairwise deletion per scenario:
      - For each scenario s, only timesteps where obs and sim[:, s] are finite
        are used for numerator, mean(obs), and denominator.
    """

    def calculate_metric(self, obs, sim):
        n_scen = sim.shape[1]
        out = np.full(n_scen, np.nan, dtype=float)

        for s in range(n_scen):
            mask = np.isfinite(obs) & np.isfinite(sim[:, s])
            if np.count_nonzero(mask) < 2:
                out[s] = np.nan
                continue

            o = obs[mask]
            m = sim[mask, s]

            o_mean = np.mean(o)
            numerator = np.sum((o - m) ** 2)
            denominator = np.sum((o - o_mean) ** 2)

            out[s] = np.nan if denominator == 0 else (1.0 - (numerator / denominator))

        return out


NashSutcliffeEfficiencyNodeRecorder.register()


class NashSutcliffeEfficiencyStorageRecorder(BaseComparisonStorageRecorder):
    """
    Nash–Sutcliffe Efficiency (NSE) metric for Storage node volumes.

    NSE = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)

    Returns per-scenario NSE values.

    Missing values
    --------------
    Pairwise deletion per scenario:
      - For each scenario s, only timesteps where obs and sim[:, s] are finite
        are used for numerator, mean(obs), and denominator.
    """

    def calculate_metric(self, obs, sim):
        n_scen = sim.shape[1]
        out = np.full(n_scen, np.nan, dtype=float)

        for s in range(n_scen):
            mask = np.isfinite(obs) & np.isfinite(sim[:, s])
            if np.count_nonzero(mask) < 2:
                out[s] = np.nan
                continue

            o = obs[mask]
            m = sim[mask, s]

            o_mean = np.mean(o)
            numerator = np.sum((o - m) ** 2)
            denominator = np.sum((o - o_mean) ** 2)

            out[s] = np.nan if denominator == 0 else (1.0 - (numerator / denominator))

        return out


NashSutcliffeEfficiencyStorageRecorder.register()


class AbsolutePercentBiasNodeRecorder(BaseComparisonNodeRecorder):
    """
    Absolute Percent Bias (|PBIAS|) metric for Node flows.

    |PBIAS| = abs( sum(obs - sim) / sum(obs) ) * 100

    Returns per-scenario absolute percent bias values.

    Missing values
    --------------
    Pairwise deletion per scenario:
      - For each scenario s, only timesteps where obs and sim[:, s] are finite
        are included in sums.
    """

    def calculate_metric(self, obs, sim):
        n_scen = sim.shape[1]
        out = np.full(n_scen, np.nan, dtype=float)

        for s in range(n_scen):
            mask = np.isfinite(obs) & np.isfinite(sim[:, s])
            if np.count_nonzero(mask) < 1:
                out[s] = np.nan
                continue

            o = obs[mask]
            m = sim[mask, s]

            sum_obs = np.sum(o)
            if sum_obs == 0:
                out[s] = np.nan
                continue

            out[s] = np.abs(np.sum(o - m) * 100.0 / sum_obs)

        return out


AbsolutePercentBiasNodeRecorder.register()


class AbsolutePercentBiasStorageRecorder(BaseComparisonStorageRecorder):
    """
    Absolute Percent Bias (|PBIAS|) metric for Storage node volumes.

    |PBIAS| = abs( sum(obs - sim) / sum(obs) ) * 100

    Returns per-scenario absolute percent bias values.

    Missing values
    --------------
    Pairwise deletion per scenario:
      - For each scenario s, only timesteps where obs and sim[:, s] are finite
        are included in sums.
    """

    def calculate_metric(self, obs, sim):
        n_scen = sim.shape[1]
        out = np.full(n_scen, np.nan, dtype=float)

        for s in range(n_scen):
            mask = np.isfinite(obs) & np.isfinite(sim[:, s])
            if np.count_nonzero(mask) < 1:
                out[s] = np.nan
                continue

            o = obs[mask]
            m = sim[mask, s]

            sum_obs = np.sum(o)
            if sum_obs == 0:
                out[s] = np.nan
                continue

            out[s] = np.abs(np.sum(o - m) * 100.0 / sum_obs)

        return out


AbsolutePercentBiasStorageRecorder.register()

class AbsolutePercentBiasStorageRecorder(BaseComparisonStorageRecorder):
    """
    Absolute Percent Bias (|PBIAS|) metric for Storage node volumes.

    |PBIAS| = abs( sum(obs - sim) / sum(obs) ) * 100

    Returns per-scenario absolute percent bias values.

    Missing values
    --------------
    Pairwise deletion per scenario:
      - For each scenario s, only timesteps where obs and sim[:, s] are finite
        are included in sums.
    """

    def calculate_metric(self, obs, sim):
        n_scen = sim.shape[1]
        out = np.full(n_scen, np.nan, dtype=float)

        for s in range(n_scen):
            mask = np.isfinite(obs) & np.isfinite(sim[:, s])
            if np.count_nonzero(mask) < 1:
                out[s] = np.nan
                continue

            o = obs[mask]
            m = sim[mask, s]

            sum_obs = np.sum(o)
            if sum_obs == 0:
                out[s] = np.nan
                continue

            out[s] = np.abs(np.sum(o - m) * 100.0 / sum_obs)

        return out


AbsolutePercentBiasStorageRecorder.register()


"""
=========================================================================
Custom performance metric recorders for irrigation revenue calibration
"""

class AnnualIrrigationRevenueRecorder(NodeRecorder):
    """
    Annual irrigation revenue per scenario derived from annual supply/demand curtailment.

    This new recorder replace the old "AverageAnnualIrrigationRevenueScenarioRecorder" which is in the run-wb-models branch.

    Purpose
    -------
    This recorder produces (1) an annual time series of irrigation revenue for each scenario,
    and (2) a single scalar value per scenario by aggregating the annual revenues over time.

    The recorder is designed for irrigation nodes where:
      - the node supply is represented by `node.flow` (per scenario), and
      - the irrigation demand is represented by the node's `max_flow` parameter (or a
        supplied `demand_parameter` override).

    The motivation for annual aggregation is to make objective functions stable across
    different model timestep granularities (e.g., monthly vs daily), provided that the
    underlying units are consistent and timestep totals are computed correctly.

    Core calculation
    ----------------
    Revenue is computed from a curtailment ratio derived on annual totals:

        annual_supply  = sum_t( supply_rate_t  * timestep_days_t )     if flow_is_per_day
        annual_demand  = sum_t( demand_rate_t  * timestep_days_t )     if flow_is_per_day
        r_y            = annual_supply / annual_demand

    Annual crop yield (kg) and annual revenue (M$) are then computed as:

        crop_yield_kg  = r_y * area_ha * yield_per_area_kg_ha
        revenue_M$     = (crop_yield_kg / 1000) * price_$/t / 1e6

    Key assumptions and required unit consistency
    ---------------------------------------------
    The recorder does NOT perform unit conversion. You must ensure that all relevant
    quantities are physically consistent:

    1) Supply and demand rates:
       - `node.flow` is assumed to be a per-day rate (flow_is_per_day=True),
         meaning it must be multiplied by `timestep.days` to obtain a timestep total.
       - `demand_parameter` (or `node.max_flow`) must be on the same per-day basis if
         flow_is_per_day=True. This matches the common pattern where irrigation demand
         is returned as Mm3/day by a parameter.

       If your model uses per-timestep totals already, set flow_is_per_day=False to prevent
       multiplying by `timestep.days`.

    2) Yield and area:
       - `area` must be in hectares (ha).
       - `yield_per_area` must be in kg/ha.

    3) Price:
       - `price` must be in dollars per metric tonne ($/t).

    4) Output units:
       - Revenue is returned in million dollars (M$).

    Curtailment ratio semantics
    ---------------------------
    Curtailment ratio r_y is computed on ANNUAL totals:

        r_y = annual_supply / annual_demand

    Special cases:
    - annual_demand == 0:
        r_y is set to `zero_demand_ratio` (default 1.0).
        Rationale: for irrigation-demand parameters (e.g., FAO-based irrigation water requirement),
        demand may legitimately be zero (e.g., rainfall meets crop water needs). In that case, zero
        irrigation requirement should not force crop yield or revenue to zero.
    - clip_ratio:
        If True, r_y is clipped to [0, 1] to prevent over-supply inflating yield beyond the
        nominal maximum.

    Pywr integration and outputs
    ----------------------------
    - The recorder maintains annual accumulators during the run (in `after()`), then computes
      final annual revenue in `finish()`.
    - `to_dataframe()` returns the full annual revenue time series as a DataFrame with:
        index   = annual timestamps (YYYY-12-31),
        columns = model.scenarios.multiindex,
        values  = annual revenue in M$.
    - `values()` returns a 1D numpy array of length n_scenarios: scalar revenue value per scenario,
      computed by applying a temporal aggregation (`temporal_agg_func`) across the annual time axis.
    - Scenario aggregation across scenarios (if required by the optimisation framework) should use
      Pywr's standard `agg_func` mechanism (e.g., mean/median/min/max) applied by the base recorder
      machinery (Recorder.aggregated_value()).

    Configuration via JSON
    ----------------------
    This recorder can be instantiated from a Pywr JSON model file. The typical configuration is:

    Example 1: Minimal configuration (use node.max_flow for demand, area/yield from demand parameter)
    -----------------------------------------------------------------------------------------------
    {
      "my_revenue_recorder": {
        "type": "AnnualIrrigationRevenueRecorder",
        "node": "AGR_TJK_Vakhsh_Cotton",
        "price": 2393,
        "temporal_agg_func": "mean",
        "agg_func": "mean"
      }
    }

    Example 2: Override area (Parameter reference) and use robust defaults
    ----------------------------------------------------------------------
    {
      "__AGR_TJK_Vakhsh_Cotton__:Average annual crop yield revenue recorder": {
        "type": "AnnualIrrigationRevenueRecorder",
        "node": "AGR_TJK_Vakhsh_Cotton",
        "agg_func": "median",
        "temporal_agg_func": "median",
        "area": "__AGR_TJK_Vakhsh_Cotton__:area",
        "price": 2393,
        "zero_demand_ratio": 1.0,
        "clip_ratio": true,
        "flow_is_per_day": true
      }
    }

    Example 3: Override demand_parameter explicitly (optional)
    ----------------------------------------------------------
    {
      "my_revenue_recorder": {
        "type": "AnnualIrrigationRevenueRecorder",
        "node": "AGR_TJK_Vakhsh_Cotton",
        "demand_parameter": "__AGR_TJK_Vakhsh_Cotton__:max_flow",
        "price": 2393
      }
    }

    JSON fields
    -----------
    Required:
      - node: node reference (string or dict compatible with model._get_node_from_ref)

    Optional:
      - price: float or Parameter reference
      - temporal_agg_func: temporal aggregator name (e.g., "mean", "median", "min", "max")
      - agg_func: scenario aggregator name used by Pywr when requesting a single aggregated value
      - flow_is_per_day: bool; if True multiply supply/demand rates by timestep.days
      - zero_demand_ratio: float; ratio used when annual_demand == 0 (default 1.0)
      - clip_ratio: bool; clip ratio to [0, 1] (default True)
      - area: float or Parameter reference; if absent uses demand_parameter.area when available
      - yield_per_area: float or Parameter reference; if absent uses demand_parameter.yield_per_area when available
      - demand_parameter: float/Parameter reference (typically a Parameter); if absent uses node.max_flow

    Lifecycle and data flow (implementation overview)
    -------------------------------------------------
    - setup():
        * Builds a mapping from each model timestep to an annual "year slot".
        * Allocates annual accumulators for supply and demand with shape (n_years, n_scenarios).
        * Resolves the demand parameter source:
            - uses `demand_parameter` if provided, otherwise uses `node.max_flow`.
        * Resolves the sources for area and yield_per_area:
            - uses explicit overrides if provided, otherwise attempts to read
              `.area` and `.yield_per_area` from the demand parameter.
        * Clears cached scenario arrays and outputs for safety.

    - after():
        * Retrieves the current timestep and its annual slot (year index).
        * Lazily evaluates scenario-constant arrays (area, yield, price) the first time `after()` runs.
          This avoids evaluating parameters too early in the model lifecycle.
        * Retrieves per-scenario supply from `node.flow`.
        * Retrieves per-scenario demand from the demand parameter (vectorised if possible; otherwise per scenario).
        * Converts rates to timestep totals using timestep.days if flow_is_per_day=True.
        * Accumulates timestep totals into annual supply/demand arrays.

    - finish():
        * Computes annual curtailment ratios from annual totals.
        * Applies the zero-demand handling rule and optional clipping.
        * Computes annual crop yield (kg) and annual revenue (M$).
        * Caches both the annual revenue numpy array and a DataFrame view.

    - to_dataframe():
        * Returns the cached annual DataFrame (computes via finish() if needed).

    - values():
        * Returns per-scenario scalar values by temporally aggregating annual revenue (axis=0).

    Notes on persistence-oriented frequency indicator
    -------------------------------------------------
    - The recorder avoids pandas resampling and instead uses an explicit year mapping derived from the
      model's timestep index. This reduces assumptions about timestep regularity (daily/monthly/etc.).
    - The `zero_demand_ratio` behaviour is critical for irrigation-demand parameters where irrigation
      demand can legitimately be zero without implying crop failure.

    """

    def __init__(
        self,
        model,
        node,
        price=1.0,
        temporal_agg_func="mean",
        flow_is_per_day=True,
        zero_demand_ratio=1.0,
        clip_ratio=True,
        area=None,
        yield_per_area=None,
        demand_parameter=None,
        **kwargs,
    ):
        """
        Initialise the recorder.

        Parameters
        ----------
        model : pywr.core.Model
            Pywr model instance.
        node : pywr.core.Node
            Node whose supply is evaluated using `node.flow`.
        price : float or Parameter, default 1.0
            Crop price in $/t. May be scenario-dependent if provided as a Parameter.
        temporal_agg_func : str, default "mean"
            Aggregation function applied over the annual time axis in `values()`
            (e.g., "mean", "median", "min", "max").
        flow_is_per_day : bool, default True
            If True, treat supply and demand as per-day rates and multiply by timestep.days
            to obtain timestep totals before annual aggregation.
        zero_demand_ratio : float, default 1.0
            Curtailment ratio used when annual demand is zero.
        clip_ratio : bool, default True
            If True, clip curtailment ratio to [0, 1].
        area : float or Parameter, optional
            Override for crop area (ha). If None, uses demand_parameter.area when available.
        yield_per_area : float or Parameter, optional
            Override for yield (kg/ha). If None, uses demand_parameter.yield_per_area when available.
        demand_parameter : Parameter, optional
            Override for irrigation demand source. If None, uses node.max_flow.

        Other Parameters
        ----------------
        **kwargs are passed to the NodeRecorder base class. This includes standard recorder
        options such as:
          - name
          - comment
          - ignore_nan
          - agg_func (scenario aggregation)
        """
        super().__init__(model, node, **kwargs)

        self.flow_is_per_day = bool(flow_is_per_day)
        self.zero_demand_ratio = float(zero_demand_ratio)
        self.clip_ratio = bool(clip_ratio)

        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

        # Optional overrides (float or Parameter). If None, pulled from node.max_flow.
        self._area_src = area
        self._yield_src = yield_per_area
        self._price_src = price
        self._demand_src = demand_parameter  # defaults to node.max_flow if None

        # Cached scenario arrays (n_scen,), evaluated lazily on first `after()`
        self._area = None
        self._yield = None
        self._price = None

        # Annual accumulators (n_years, n_scen)
        self._year_labels = None
        self._year_index = None
        self._annual_supply = None
        self._annual_demand = None

        # Cached final annual revenue
        self._annual_revenue = None
        self._annual_df = None

        # Demand parameter resolved in setup
        self._demand_param = None
        self._resolved_area_src = None
        self._resolved_yield_src = None

    @staticmethod
    def _eval_scenario_array(model, src, n_scen):
        """
        Evaluate a float/Parameter into a (n_scen,) array.

        Why evaluation is delayed
        -------------------------
        Parameters in Pywr may depend on internal runtime state that is fully initialised
        only once the model is actively stepping through timesteps. Evaluating scenario arrays
        lazily (during/after the first `after()` call) avoids accessing parameter internals too early.

        Behaviour
        ---------
        - If src is a numeric scalar, returns a constant vector of length n_scen.
        - If src exposes get_all_values(), attempts vectorised evaluation.
        - Otherwise, falls back to calling get_value(scenario) for each scenario.

        Parameters
        ----------
        model : pywr.core.Model
            Pywr model.
        src : float or pywr.parameters.Parameter
            Value source.
        n_scen : int
            Number of scenario combinations.

        Returns
        -------
        numpy.ndarray
            A 1D array (n_scen,) containing the value for each scenario.
        """
        if isinstance(src, (int, float, np.floating)):
            return np.full(n_scen, float(src), dtype=float)

        # Prefer vectorised evaluation; if it fails, fall back to per-scenario loop.
        if hasattr(src, "get_all_values"):
            try:
                vals = np.asarray(src.get_all_values(), dtype=float)
                if vals.shape[0] == n_scen:
                    return vals
            except Exception:
                pass

        out = np.zeros(n_scen, dtype=float)
        for si in model.scenarios.combinations:
            out[si.global_id] = float(src.get_value(si))
        return out

    def setup(self):
        """
        Allocate annual accumulators and resolve parameter sources.

        This method:
          1) Builds an array mapping each timestep index to its year slot.
          2) Allocates annual supply/demand accumulators: shape (n_years, n_scenarios).
          3) Resolves the demand parameter:
               - uses explicit demand_parameter override if provided,
               - otherwise uses node.max_flow.
          4) Resolves the sources for area and yield_per_area (but does not evaluate them yet).
          5) Clears cached arrays and output caches.

        Raises
        ------
        AttributeError
            If demand cannot be resolved (no max_flow and no demand_parameter provided),
            or if area/yield cannot be resolved from overrides or the demand parameter.
        """
        super().setup()

        dt_index = self.model.timestepper.datetime_index
        if isinstance(dt_index, pd.PeriodIndex):
            dt_index = dt_index.to_timestamp()

        years = np.asarray([d.year for d in dt_index], dtype=int)
        self._year_labels = np.unique(years)
        year_to_slot = {y: i for i, y in enumerate(self._year_labels)}
        self._year_index = np.asarray([year_to_slot[y] for y in years], dtype=int)

        n_years = len(self._year_labels)
        n_scen = len(self.model.scenarios.combinations)

        self._annual_supply = np.zeros((n_years, n_scen), dtype=float)
        self._annual_demand = np.zeros((n_years, n_scen), dtype=float)

        # Resolve demand parameter (defaults to node.max_flow)
        if self._demand_src is not None:
            self._demand_param = self._demand_src
        else:
            mf = getattr(self.node, "max_flow", None)
            if mf is None:
                raise AttributeError("Node has no max_flow; provide demand_parameter in recorder config.")
            self._demand_param = mf

        # Resolve area/yield sources (but do NOT evaluate to arrays yet)
        self._resolved_area_src = self._area_src if self._area_src is not None else getattr(self._demand_param, "area", None)
        self._resolved_yield_src = self._yield_src if self._yield_src is not None else getattr(self._demand_param, "yield_per_area", None)

        if self._resolved_area_src is None or self._resolved_yield_src is None:
            raise AttributeError("Could not resolve area and/or yield_per_area for revenue calculation.")

        # Reset cached arrays/results
        self._area = None
        self._yield = None
        self._price = None
        self._annual_revenue = None
        self._annual_df = None

    def reset(self):
        """
        Reset annual accumulators and cached outputs for a new run.

        Notes
        -----
        This resets the annual supply/demand totals and clears cached outputs.
        It also clears cached scenario arrays (area, yield, price) to ensure correct
        behaviour if the model is reused for multiple runs.
        """
        super().reset()
        self._annual_supply[:, :] = 0.0
        self._annual_demand[:, :] = 0.0
        self._annual_revenue = None
        self._annual_df = None

        # Keep _area/_yield/_price cached across reset? Safer to clear for multi-run usage.
        self._area = None
        self._yield = None
        self._price = None

    def after(self):
        """
        Accumulate timestep supply/demand totals into annual totals for each scenario.

        Steps
        -----
        1) Determine the year slot for the current timestep.
        2) Lazily evaluate scenario-constant arrays for area, yield, and price on the first call.
        3) Retrieve per-scenario supply rate from `node.flow`.
        4) Retrieve per-scenario demand rate from the resolved demand parameter:
             - use get_all_values() when available; otherwise evaluate per scenario.
        5) Convert rates to timestep totals via timestep.days if flow_is_per_day=True.
        6) Accumulate into annual totals.

        Returns
        -------
        None
            Pywr recorder hooks typically return None; the original code returns nothing explicitly.
        """
        ts = self.model.timestepper.current
        y_i = self._year_index[ts.index]

        n_scen = self._annual_supply.shape[1]

        # Lazily evaluate scenario-constant arrays the first time a timestep exists
        if self._area is None:
            self._area = self._eval_scenario_array(self.model, self._resolved_area_src, n_scen)
        if self._yield is None:
            self._yield = self._eval_scenario_array(self.model, self._resolved_yield_src, n_scen)
        if self._price is None:
            self._price = self._eval_scenario_array(self.model, self._price_src, n_scen)

        w = float(ts.days) if self.flow_is_per_day else 1.0

        # Supply: node.flow is per scenario
        supply_rate = np.asarray(self.node.flow, dtype=float)
        if supply_rate.shape == ():  # scalar safeguard
            supply_rate = np.full(n_scen, float(supply_rate), dtype=float)

        # Demand: prefer get_all_values if available, otherwise loop
        if hasattr(self._demand_param, "get_all_values"):
            try:
                demand_rate = np.asarray(self._demand_param.get_all_values(), dtype=float)
            except Exception:
                demand_rate = np.zeros(n_scen, dtype=float)
                for si in self.model.scenarios.combinations:
                    demand_rate[si.global_id] = float(self._demand_param.get_value(si))
        else:
            demand_rate = np.zeros(n_scen, dtype=float)
            for si in self.model.scenarios.combinations:
                demand_rate[si.global_id] = float(self._demand_param.get_value(si))

        # Convert per-day rates to timestep totals
        self._annual_supply[y_i, :] += supply_rate * w
        self._annual_demand[y_i, :] += demand_rate * w

    def finish(self):
        """
        Compute annual revenue arrays and cache the outputs.

        This method:
          1) Validates that scenario-constant inputs were evaluated.
          2) Computes annual curtailment ratio using annual totals.
          3) Applies the zero-demand rule and handles NaN/inf safely.
          4) Optionally clips ratio to [0, 1].
          5) Computes annual crop yield (kg) and annual revenue (M$).
          6) Builds a DataFrame view indexed by year end.

        Raises
        ------
        RuntimeError
            If area/yield/price were not evaluated. This indicates `after()` did not run,
            typically because the model did not execute timesteps.
        """
        super().finish()

        if self._area is None or self._yield is None or self._price is None:
            raise RuntimeError(
                "Scenario-constant inputs (area/yield/price) were not evaluated. "
                "This typically means `after()` was never called (e.g., model did not run)."
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                self._annual_demand == 0.0,
                self.zero_demand_ratio,
                self._annual_supply / self._annual_demand,
            )
            ratio = np.nan_to_num(ratio, nan=self.zero_demand_ratio, posinf=self.zero_demand_ratio, neginf=0.0)

        if self.clip_ratio:
            ratio = np.clip(ratio, 0.0, 1.0)

        # crop_yield_kg (years x scenarios), then revenue in M$
        crop_yield_kg = ratio * self._area[None, :] * self._yield[None, :]
        revenue = (crop_yield_kg / 1e3) * self._price[None, :] / 1e6

        self._annual_revenue = revenue

        sc_index = self.model.scenarios.multiindex
        annual_dt_index = pd.to_datetime([f"{y}-12-31" for y in self._year_labels])
        self._annual_df = pd.DataFrame(revenue, index=annual_dt_index, columns=sc_index)

    def to_dataframe(self):
        """
        Return annual irrigation revenue time series (M$) for each scenario.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by year-end timestamps with scenario MultiIndex columns.
            Values are annual revenues in M$.
        """
        if self._annual_df is None:
            self.finish()
        return self._annual_df

    def values(self):
        """
        Return per-scenario scalar by aggregating annual revenue over years.

        The aggregation is performed using Pywr's Aggregator on the annual revenue
        array produced in `finish()`.

        Returns
        -------
        numpy.ndarray
            1D array with length equal to the number of scenario combinations.
        """
        if self._annual_revenue is None:
            self.finish()

        return self._temporal_aggregator.aggregate_2d(
            self._annual_revenue,
            axis=0,  # aggregate over years
            ignore_nan=self.ignore_nan,
        )

    @classmethod
    def load(cls, model, data):
        """
        Create an AnnualIrrigationRevenueRecorder from a Pywr JSON definition.

        Parameters
        ----------
        model : pywr.core.Model
            Model instance.
        data : dict
            Parsed JSON dictionary for this recorder.

        Supported JSON keys
        -------------------
        - node (required): node reference for which supply is read from node.flow
        - price: float or Parameter reference (default 1.0)
        - temporal_agg_func: str (default "mean")
        - flow_is_per_day: bool (default True)
        - zero_demand_ratio: float (default 1.0)
        - clip_ratio: bool (default True)
        - area: float or Parameter reference (optional override)
        - yield_per_area: float or Parameter reference (optional override)
        - demand_parameter: Parameter reference (optional override; default uses node.max_flow)

        Notes
        -----
        Parameter references are resolved using `pywr.parameters.load_parameter`.
        """
        node = model._get_node_from_ref(model, data.pop("node"))

        def load_float_or_param(key, default=None):
            if key not in data:
                return default
            v = data.pop(key)
            if isinstance(v, (int, float)):
                return float(v)
            return load_parameter(model, v)

        return cls(
            model,
            node,
            price=load_float_or_param("price", 1.0),
            temporal_agg_func=data.pop("temporal_agg_func", "mean"),
            flow_is_per_day=data.pop("flow_is_per_day", True),
            zero_demand_ratio=data.pop("zero_demand_ratio", 1.0),
            clip_ratio=data.pop("clip_ratio", True),
            area=load_float_or_param("area", None),
            yield_per_area=load_float_or_param("yield_per_area", None),
            demand_parameter=load_float_or_param("demand_parameter", None),
            **data,
        )


AnnualIrrigationRevenueRecorder.register()


class AnnualHydroPowerRecorder(NodeRecorder):
    """
    Annual hydropower (energy) or hydropower-derived revenue recorder with optional seasonal month selection.

    Overview
    --------
    This recorder computes annual totals per scenario using Pywr's hydropower calculation. It is designed
    to be robust for non-daily timesteps (e.g., "7D") and avoids the common pattern of:
        timestep values -> daily resample/ffill -> month filter -> annual resample
    which can introduce allocation errors when timesteps straddle month boundaries.

    Instead, this recorder:
      1) Computes an energy-rate-per-day for each timestep using `hydropower_calculation`.
      2) Converts to energy over the timestep by multiplying by the number of days in the timestep.
      3) If a month-season filter is defined, splits each timestep at month boundaries and allocates energy
         by exact day fractions.
      4) Aggregates into annual totals per scenario.

    Crucially: removing the "zero year" completely
    ----------------------------------------------
    If you define a timestepper like:
        start = 2036-01-01
        end   = 2064-12-31
        timestep = 7D
    it is easy for custom annual recorders to accidentally include an extra year row (e.g., 2065) if they
    infer the year range from the last timestep end, rather than the configured end.

    This implementation uses the model timestepper definition directly:
      - Annual years are capped to [start.year, end.year] inclusive.
      - The last timestep is clipped so no contribution is counted after the configured model end date.

    That ensures you never get an "extra" year of all zeros from year-range inference.

    Head definition
    ---------------
    The head used in hydropower is computed as:
        head = water_elevation - turbine_elevation
    when `water_elevation_parameter` is provided. Negative head is clipped to zero.
    If no water elevation parameter is provided, `turbine_elevation` is treated as the head directly.

    Energy and revenue scaling (using `factor` as in Pywr)
    ------------------------------------------------------
    Many Pywr workflows apply a multiplicative scaling to recorder outputs. In some Pywr builds, `factor`
    is not accepted by `NodeRecorder.__init__`. Therefore, this recorder:
      - Pops `factor` from kwargs before calling `super().__init__`
      - Applies `factor` once, at the end, to annual totals.

    This supports the common pattern:
        revenue = energy * price
    where `factor` represents a price or conversion (e.g., M$ per MWh).

    Seasonal month selection
    ------------------------
    If `monthly_seasonality` is provided (list of months 1..12), only the portion of each timestep that
    lies within those months is included. This is done by splitting each timestep at calendar month
    boundaries and allocating energy by exact day fractions.

    JSON usage (general example)
    ----------------------------
    The recorder can be used directly from a Pywr JSON model. The following is a general example
    (names are placeholders):

    {
      "__TURBINE_NODE__:Annual Revenue": {
        "type": "AnnualHydroPowerRecorder",
        "node": "TURBINE_NODE_NAME",
        "agg_func": "mean",
        "temporal_agg_func": "mean",
        "efficiency": 0.95,
        "turbine_elevation": 100.0,
        "water_elevation_parameter": "RESERVOIR_LEVEL_PARAMETER",
        "flow_unit_conversion": 11.57407407,
        "energy_unit_conversion": 2.4e-05,
        "factor": 4.2e-05,
        "monthly_seasonality": [4, 5, 6, 7, 8, 9]
      }
    }

    Notes on conversions:
      - This recorder does not enforce a single unit system; it assumes your chosen conversion constants
        are consistent with your model's flow units and desired energy units.
      - With flow in million m³/day (Mm³/day), the pair:
            flow_unit_conversion   = 11.57407407  (= 1e6 / 86400)   -> Mm³/day to m³/s
            energy_unit_conversion = 2.4e-05      (= 86400 / 3.6e9) -> W to MWh/day
        is internally consistent for an output in MWh/day, which is then multiplied by days to give MWh.

    Outputs
    -------
    - to_dataframe(): annual totals (years x scenarios)
    - values(): per-scenario scalar from aggregating annual totals across years via `temporal_agg_func`

    """

    def __init__(
        self,
        model,
        node,
        monthly_seasonality=None,
        water_elevation_parameter=None,
        turbine_elevation=0.0,
        efficiency=1.0,
        density=1000.0,
        flow_unit_conversion=1.0,
        energy_unit_conversion=1e-6,
        **kwargs,
    ):
        # Pywr build compatibility: `factor` may not be accepted by NodeRecorder/Component __init__.
        # We consume it here and apply it later in `finish()`.
        self.factor = kwargs.pop("factor", None)

        # Temporal aggregation (over years) for values()
        temporal_agg_func = kwargs.pop("temporal_agg_func", "mean")

        # Initialise base recorder (handles name/comment/agg_func/ignore_nan/etc.)
        super().__init__(model, node, **kwargs)

        # Store temporal aggregator for values()
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

        # Optional month filtering
        self._monthly_seasonality = monthly_seasonality

        # Optional elevation parameter used to compute head
        self._water_elevation_parameter = None
        self.water_elevation_parameter = water_elevation_parameter

        # Hydropower inputs
        self.turbine_elevation = float(turbine_elevation)
        self.efficiency = float(efficiency)
        self.density = float(density)
        self.flow_unit_conversion = float(flow_unit_conversion)
        self.energy_unit_conversion = float(energy_unit_conversion)

        # Model period (derived from timestepper in setup())
        self._model_start = None            # pd.Timestamp
        self._model_end = None              # pd.Timestamp (date)
        self._model_stop = None             # pd.Timestamp = model_end + 1 day (exclusive upper bound)

        # Timeline (timestep start timestamps)
        self._dt_starts = None              # list[pd.Timestamp]

        # Annual indexing
        self._year_labels = None            # np.ndarray of years
        self._year_to_slot = None           # dict year -> row index

        # Accumulated annual totals, and cached DataFrame view
        self._annual_totals = None          # shape (n_years, n_scen)
        self._annual_df = None              # DataFrame after finish()

        # Cache whether hydropower_calculation supports vectorised inputs in this environment
        self._hydro_vectorized = None

    @property
    def water_elevation_parameter(self):
        """Optional Parameter providing upstream water elevation used to compute head."""
        return self._water_elevation_parameter

    @water_elevation_parameter.setter
    def water_elevation_parameter(self, parameter):
        """
        Set water elevation parameter and register it as a dependency (child) so Pywr evaluates it correctly.
        """
        current = getattr(self, "_water_elevation_parameter", None)
        if current is not None and current in self.children:
            self.children.remove(current)
        if parameter is not None:
            self.children.add(parameter)
        self._water_elevation_parameter = parameter

    def setup(self):
        """
        Allocate arrays and build year indexing.

        The year range is capped using the model's timestepper definition:
            years = start.year .. end.year (inclusive)
        This prevents creation of an "extra" year row beyond the configured model end.
        """
        super().setup()

        # Read the model period from timestepper (preferred), otherwise fall back to the datetime_index.
        ts_start = getattr(self.model.timestepper, "start", None)
        ts_end = getattr(self.model.timestepper, "end", None)

        dt_index = self.model.timestepper.datetime_index
        if isinstance(dt_index, pd.PeriodIndex):
            dt_index = dt_index.to_timestamp()

        self._dt_starts = [pd.Timestamp(d) for d in dt_index]

        if ts_start is not None and ts_end is not None:
            self._model_start = pd.Timestamp(ts_start)
            self._model_end = pd.Timestamp(ts_end)
        else:
            # Fallback: infer from datetime_index; this is less precise than using explicit start/end.
            self._model_start = self._dt_starts[0]
            self._model_end = self._dt_starts[-1]

        # Use an exclusive upper bound for clipping: [start, end+1day)
        # This treats the JSON "end" as an inclusive date.
        self._model_stop = self._model_end + pd.Timedelta(days=1)

        # Build year labels strictly from the configured model period
        start_year = int(self._model_start.year)
        end_year = int(self._model_end.year)

        self._year_labels = np.arange(start_year, end_year + 1, dtype=int)
        self._year_to_slot = {y: i for i, y in enumerate(self._year_labels)}

        # Allocate annual totals array
        n_years = len(self._year_labels)
        n_scen = len(self.model.scenarios.combinations)
        self._annual_totals = np.zeros((n_years, n_scen), dtype=float)

        # Reset caches
        self._annual_df = None
        self._hydro_vectorized = None

    def reset(self):
        """Reset annual totals for a new model run."""
        super().reset()
        self._annual_totals[:, :] = 0.0
        self._annual_df = None

    def _iter_month_segments(self, dt_start, dt_end):
        """
        Yield (segment_start, segment_end) pairs that do not cross calendar month boundaries.

        This is the mechanism that makes seasonal slicing correct for 7D (or other) timesteps.
        """
        cur = dt_start
        while cur < dt_end:
            # First moment of next month
            next_month = (cur.to_period("M") + 1).to_timestamp()
            seg_end = next_month if next_month < dt_end else dt_end
            yield cur, seg_end
            cur = seg_end

    @staticmethod
    def _segment_days(seg_start, seg_end):
        """Return segment length in days (float)."""
        return (seg_end - seg_start).total_seconds() / 86400.0

    def _water_elevation_values(self, n_scen):
        p = self._water_elevation_parameter
        if p is None:
            return None

        if isinstance(p, (int, float, np.floating)):
            return np.full(n_scen, float(p), dtype=float)

        if hasattr(p, "get_all_values"):
            try:
                vals = np.asarray(p.get_all_values(), dtype=float)
                if vals.shape[0] == n_scen:
                    return vals
            except Exception:
                pass

        out = np.zeros(n_scen, dtype=float)
        for si in self.model.scenarios.combinations:
            out[si.global_id] = float(p.get_value(si))
        return out

    def _hydropower_rate(self, q, head):
        """
        Compute hydropower output using Pywr's `hydropower_calculation`.

        Attempts vectorised evaluation for performance; falls back to per-scenario if required.
        The returned values are treated as an energy rate per day in units implied by
        `energy_unit_conversion`.
        """
        if self._hydro_vectorized is False:
            out = np.zeros_like(q, dtype=float)
            for i in range(q.shape[0]):
                out[i] = hydropower_calculation(
                    float(q[i]),
                    float(head[i]),
                    0.0,
                    self.efficiency,
                    density=self.density,
                    flow_unit_conversion=self.flow_unit_conversion,
                    energy_unit_conversion=self.energy_unit_conversion,
                )
            return out

        try:
            out = hydropower_calculation(
                q,
                head,
                0.0,
                self.efficiency,
                density=self.density,
                flow_unit_conversion=self.flow_unit_conversion,
                energy_unit_conversion=self.energy_unit_conversion,
            )
            self._hydro_vectorized = True
            return np.asarray(out, dtype=float)
        except Exception:
            self._hydro_vectorized = False
            return self._hydropower_rate(q, head)

    def after(self):
        """
        Accumulate annual totals for the current timestep.

        Steps:
          1) Determine timestep [dt_start, dt_end_raw)
          2) Clip dt_end to the configured model stop date (end+1day) to avoid counting beyond end.
          3) Compute head and hydropower rate per day for each scenario.
          4) Split timestep into month segments; optionally filter by month.
          5) Add energy_per_day * segment_days to the corresponding annual bucket.
        """
        ts = self.model.timestepper.current
        t_i = ts.index

        dt_start = self._dt_starts[t_i]
        dt_end_raw = dt_start + pd.Timedelta(days=float(ts.days))

        # Clip to model period to ensure we do not count beyond configured end date.
        dt_end = dt_end_raw if dt_end_raw <= self._model_stop else self._model_stop
        if dt_end <= dt_start:
            return

        n_scen = len(self.model.scenarios.combinations)

        # Model flow is a per-day rate in the model's flow units (your case: Mm³/day).
        q = np.asarray(self.node.flow, dtype=float)
        if q.shape == ():
            q = np.full(n_scen, float(q), dtype=float)

        # Compute head per scenario
        water_elev = self._water_elevation_values(n_scen)
        if water_elev is None:
            head = np.full(n_scen, self.turbine_elevation, dtype=float)
        else:
            head = water_elev - self.turbine_elevation
        head = np.maximum(head, 0.0)

        # Compute energy rate per day (units implied by energy_unit_conversion)
        energy_per_day = self._hydropower_rate(q, head)

        # Month filter, if provided
        months_set = None
        if self._monthly_seasonality is not None:
            months_set = set(int(m) for m in self._monthly_seasonality)

        # Split and allocate the timestep by month boundary to avoid season bias with 7D timesteps
        for seg_start, seg_end in self._iter_month_segments(dt_start, dt_end):
            if months_set is not None and seg_start.month not in months_set:
                continue

            seg_days = self._segment_days(seg_start, seg_end)
            if seg_days <= 0:
                continue

            # Allocate to the year of the segment start. (Month segments do not cross year boundaries.)
            year = int(seg_start.year)
            year_slot = self._year_to_slot.get(year, None)
            if year_slot is None:
                # Outside the configured year range; ignore
                continue

            self._annual_totals[year_slot, :] += energy_per_day * seg_days

    def finish(self):
        """
        Finalise outputs:
          - Apply `factor` if provided.
          - Build an annual DataFrame indexed by year-end (YYYY-12-31) timestamps.
        """
        super().finish()

        annual = self._annual_totals
        if self.factor is not None:
            annual = annual * float(self.factor)

        sc_index = self.model.scenarios.multiindex
        annual_dt_index = pd.to_datetime([f"{y}-12-31" for y in self._year_labels])
        self._annual_df = pd.DataFrame(annual, index=annual_dt_index, columns=sc_index)

    def to_dataframe(self):
        """Return annual totals as a DataFrame (years x scenarios)."""
        if self._annual_df is None:
            self.finish()
        return self._annual_df

    def values(self):
        """
        Return a 1D array (n_scen,) by aggregating annual totals over years using `temporal_agg_func`.
        """
        if self._annual_df is None:
            self.finish()

        return self._temporal_aggregator.aggregate_2d(
            self._annual_df.values,
            axis=0,  # aggregate over years
            ignore_nan=self.ignore_nan,
        )

    @classmethod
    def load(cls, model, data):
        """
        Load the recorder from JSON.

        Supported keys (subset; standard Recorder keys also allowed):
          - node (required)
          - monthly_seasonality (optional)
          - water_elevation_parameter (optional)
          - turbine_elevation, efficiency, density, flow_unit_conversion, energy_unit_conversion (optional)
          - factor (optional; consumed by this class and applied in finish())
          - temporal_agg_func (optional)
        """
        node = model._get_node_from_ref(model, data.pop("node"))
        monthly_seasonality = data.pop("monthly_seasonality", None)

        wep_data = data.pop("water_elevation_parameter", None)
        water_elevation_parameter = load_parameter(model, wep_data) if wep_data is not None else None

        return cls(
            model,
            node,
            monthly_seasonality=monthly_seasonality,
            water_elevation_parameter=water_elevation_parameter,
            **data,
        )


AnnualHydroPowerRecorder.register()

class SeasonalTransferConstraintRecorder(NodeRecorder):
    """
    Calculates annual seasonal transfer constraint:

        constraint = (rule - annual_target) - outflow

    where annual_target can be:
      - a scalar, e.g. 4200
      - a dict keyed by year, e.g. {2036: 4200, 2037: 4300}
      - a list/array with one value per model year
    """

    def __init__(
        self,
        model,
        node,
        node_rule,
        monthly_seasonality=None,
        annual_target=4200.0,
        **kwargs,
    ):
        temporal_agg_func = kwargs.pop("temporal_agg_func", "mean")

        super().__init__(model, node, **kwargs)
        self.node_rule = node_rule
        self.monthly_seasonality = monthly_seasonality
        self.annual_target = annual_target

        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))
        self._data_rule = np.zeros((nts, ncomb))

    def reset(self):
        super().reset()
        self._data[:, :] = 0.0
        self._data_rule[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node
        node_rule = self.node_rule

        for scenario_index in self.model.scenarios.combinations:
            gid = scenario_index.global_id
            self._data[ts.index, gid] = node.flow[gid]
            self._data_rule[ts.index, gid] = node_rule.flow[gid]

        return 0

    def _annual_sum(self, data_array):
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        df = pd.DataFrame(np.array(data_array), index=index, columns=sc_index)
        df = df.resample("D").ffill()

        if self.monthly_seasonality is not None:
            df = df[df.index.month.isin(self.monthly_seasonality)]

        last_year = index[-1].year
        df = df.loc[:str(last_year), :].resample("Y").sum()

        return df

    def _build_target_and_filter_years(self, annual_index, columns):
        """
        Returns:
            filtered_index : DatetimeIndex
            target_df      : DataFrame with same index/columns shape as filtered annual data

        Behaviour:
        - scalar target: applies to all years
        - dict target: uses only years explicitly provided in dict
        - list/array: applies one value per annual row
        """
        n_scen = len(columns)

        # Case 1: scalar target for all years
        if isinstance(self.annual_target, (int, float, np.floating)):
            arr = np.full((len(annual_index), n_scen), float(self.annual_target), dtype=float)
            target_df = pd.DataFrame(arr, index=annual_index, columns=columns)
            return annual_index, target_df

        # Case 2: dict keyed by year -> USE ONLY PROVIDED YEARS
        if isinstance(self.annual_target, dict):
            target_map = {int(k): float(v) for k, v in self.annual_target.items()}

            mask = np.array([int(dt.year) in target_map for dt in annual_index], dtype=bool)
            filtered_index = annual_index[mask]

            if len(filtered_index) == 0:
                raise ValueError(
                    "annual_target dict does not match any years in the model output."
                )

            year_values = np.array([target_map[int(dt.year)] for dt in filtered_index], dtype=float)
            arr = np.repeat(year_values[:, None], n_scen, axis=1)
            target_df = pd.DataFrame(arr, index=filtered_index, columns=columns)
            return filtered_index, target_df

        # Case 3: list/array, one value per annual row
        if isinstance(self.annual_target, (list, tuple, np.ndarray)):
            vals = np.asarray(self.annual_target, dtype=float)
            if vals.shape[0] != len(annual_index):
                raise ValueError(
                    f"annual_target length {vals.shape[0]} does not match number of years {len(annual_index)}"
                )

            arr = np.repeat(vals[:, None], n_scen, axis=1)
            target_df = pd.DataFrame(arr, index=annual_index, columns=columns)
            return annual_index, target_df

        raise TypeError(
            "annual_target must be a scalar, dict keyed by year, or list/array of yearly values."
        )

    def to_dataframe(self):
        outflow = self._annual_sum(self._data)
        rule = self._annual_sum(self._data_rule)

        filtered_index, target = self._build_target_and_filter_years(rule.index, rule.columns)

        outflow = outflow.loc[filtered_index, :]
        rule = rule.loc[filtered_index, :]

        constraint = ((rule - target) - outflow).abs()
        constraint = constraint.replace([np.inf, -np.inf], np.nan).dropna(how="all")

        # Force DataFrame output
        if isinstance(constraint, pd.Series):
            constraint = constraint.to_frame().T

        return constraint

    def values(self):
        constraint = self.to_dataframe()
        n_scen = len(self.model.scenarios.combinations)

        # Force constraint to be a DataFrame aligned correctly and safely
        if isinstance(constraint, pd.Series):
            # If squeezed to a series, determine if it squeezed the year axis or scenario axis
            if len(constraint) == n_scen:
                constraint = constraint.to_frame().T  # shape: (1, n_scen)
            else:
                constraint = constraint.to_frame()    # shape: (years, 1)

        if len(constraint) == 0:
            return np.zeros(n_scen, dtype=np.float64)

        # Read the configured temporal aggregation function (e.g., "mean")
        agg_func = getattr(self._temporal_aggregator, "func", "mean")
        if not isinstance(agg_func, str):
            agg_func = "mean"

        # Bypass the Cython aggregator: Evaluate aggregation over years natively in Pandas
        if hasattr(constraint, agg_func):
            res = getattr(constraint, agg_func)(axis=0)
        else:
            res = constraint.agg(agg_func, axis=0)

        # Ensure we return a strict 1-dimensional float64 array of shape (n_scenarios,)
        vals = np.asarray(res, dtype=np.float64)
        
        # Guard against zero-dimension scalar outputs in 1-scenario edge cases
        if vals.ndim == 0:
            vals = np.array([vals], dtype=np.float64)
        
        return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def load(cls, model, data):
        """Loads the recorder from a dictionary configuration."""

        node_name = data.pop("node")
        node_rule_name = data.pop("node_rule", None)

        node = model.nodes[node_name]
        node_rule = model.nodes[node_rule_name]

        monthly_seasonality = data.pop("monthly_seasonality", None)
        annual_target = data.pop("annual_target", 4200.0)

        return cls(
            model,
            node,
            node_rule,
            monthly_seasonality=monthly_seasonality,
            annual_target=annual_target,
            **data,
        )

SeasonalTransferConstraintRecorder.register()


class StorageTargetRecorder(NodeRecorder):
    """
    Recorder for reservoir storage target tracking.

    Records:
      - actual storage volume
      - target storage returned by a Parameter
      - signed deviation = actual - target

    `values()` returns an aggregated per-scenario metric based on the selected
    metric and temporal aggregation.

    Parameters
    ----------
    node : Storage node
        Reservoir node whose volume is recorded.
    target_parameter : Parameter
        Typically SeasonalStorageTargetParameter.
    months : list[int], optional
        Restrict evaluation to selected months, e.g. [4,5,6,7,8,9].
    metric : {"signed", "absolute", "squared", "rmse"}, default "absolute"
        Metric computed from deviation.
    annual_agg_func : {"mean", "sum", "max", "min"}, default "mean"
        Aggregation within year after daily expansion / month filtering.
    temporal_agg_func : str, default "mean"
        Aggregation across years / periods, using Pywr Aggregator.
    """

    def __init__(
        self,
        model,
        node,
        target_parameter,
        months=None,
        metric="absolute",
        annual_agg_func="mean",
        **kwargs,
    ):
        temporal_agg_func = kwargs.pop("temporal_agg_func", "mean")

        super().__init__(model, node, **kwargs)

        self.target_parameter = target_parameter
        self.months = months
        self.metric = metric.lower()
        self.annual_agg_func = annual_agg_func.lower()

        if isinstance(target_parameter, Parameter):
            self.children.add(target_parameter)

        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

        self._actual = None
        self._target = None

    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._actual = np.zeros((nts, ncomb), dtype=np.float64)
        self._target = np.zeros((nts, ncomb), dtype=np.float64)

    def reset(self):
        super().reset()
        self._actual[:, :] = 0.0
        self._target[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            gid = scenario_index.global_id
            self._actual[ts.index, gid] = node.volume[gid]
            self._target[ts.index, gid] = self.target_parameter.get_value(scenario_index)

        return 0

    def _build_series(self):
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        actual = pd.DataFrame(np.array(self._actual), index=index, columns=sc_index)
        target = pd.DataFrame(np.array(self._target), index=index, columns=sc_index)

        # Follow the same broad pattern used in your seasonal recorders:
        # daily forward-fill, then optional seasonal filtering.
        actual = actual.resample("D").ffill()
        target = target.resample("D").ffill()

        if self.months is not None:
            actual = actual[actual.index.month.isin(self.months)]
            target = target[target.index.month.isin(self.months)]

        last_year = actual.index[-1].year
        actual = actual.loc[:str(last_year), :]
        target = target.loc[:str(last_year), :]

        deviation = actual - target
        return actual, target, deviation

    def _metric_dataframe(self, deviation):
        if self.metric in ("signed", "raw", "bias"):
            out = deviation
        elif self.metric in ("absolute", "abs", "mae"):
            out = deviation.abs()
        elif self.metric in ("squared", "square", "mse", "rmse"):
            out = deviation ** 2
        else:
            raise ValueError(
                f"Unknown metric '{self.metric}'. "
                "Use one of: signed, absolute, squared, rmse."
            )

        if self.annual_agg_func == "mean":
            out = out.resample("Y").mean()
        elif self.annual_agg_func == "sum":
            out = out.resample("Y").sum()
        elif self.annual_agg_func == "max":
            out = out.resample("Y").max()
        elif self.annual_agg_func == "min":
            out = out.resample("Y").min()
        else:
            raise ValueError(
                f"Unknown annual_agg_func '{self.annual_agg_func}'. "
                "Use one of: mean, sum, max, min."
            )

        if self.metric == "rmse":
            out = np.sqrt(out)

        return out

    def to_dataframe(self):
        actual, target, deviation = self._build_series()

        # Return all three for inspection
        return pd.concat(
            {
                "actual_storage": actual,
                "target_storage": target,
                "deviation": deviation,
            },
            axis=1,
        )

    def values(self):
        _, _, deviation = self._build_series()
        metric_df = self._metric_dataframe(deviation)

        return self._temporal_aggregator.aggregate_2d(
            metric_df.values,
            axis=0,
            ignore_nan=self.ignore_nan,
        )

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        target_parameter = load_parameter(model, data.pop("target_parameter"))

        return cls(
            model,
            node=node,
            target_parameter=target_parameter,
            months=data.pop("months", None),
            metric=data.pop("metric", "absolute"),
            annual_agg_func=data.pop("annual_agg_func", "mean"),
            **data,
        )


StorageTargetRecorder.register()
