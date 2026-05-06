import numpy as np
import pandas as pd

from pywr.recorders import *
from pywr.nodes import Storage
from scipy.interpolate import Rbf
from pywr.parameter_property import parameter_property
from pywr.parameters import Parameter, load_parameter, load_parameter_values, IndexParameter


class RectifierParameter(Parameter):

    def __init__(self, model, value, lower_bounds=0.0, upper_bounds=np.inf, **kwargs):
        super(RectifierParameter, self).__init__(model, **kwargs)
        self._value = value
        self.double_size = 1
        self.integer_size = 0
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def calc_values(self, timestep):
        # constant parameter can just set the entire array to one value
        if self._value < 0.0:
            self.__values[...] =  0.0
        else:
            self.__values[...] = (self._upper_bounds -  self._lower_bounds) * self._value + self._lower_bounds

    def value(self, ts, scenario_index):
        return self._value

    def set_double_variables(self, values):
        self._value = values[0]

    def get_double_variables(self):
        return np.array([self._value, ], dtype=np.float64)

    def get_double_lower_bounds(self):
        return np.array([-0.75], dtype=np.float64)

    def get_double_upper_bounds(self):
        return np.array([1.0], dtype=np.float64)

    @classmethod
    def load(cls, model, data):
        if "value" in data:
            value = data.pop("value")
        else:
            value = load_parameter_values(model, data)
        parameter = cls(model, value, **data)
        return parameter


RectifierParameter.register()


class IndexVariableParameter(IndexParameter):

    def __init__(self, model, value, lower_bounds=0, upper_bounds=1, **kwargs):
        super(IndexVariableParameter, self).__init__(model, **kwargs)
        self._value = round(value)
        self.integer_size = 1
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def calc_values(self, timestep):
        # constant parameter can just set the entire array to one value
        self.__indices[...] = self._value
        self.__values[...] = self._value

    def set_integer_variables(self, values):
        self._value  = values[0]

    def get_integer_variables(self):
        return np.array([self._value, ], dtype=np.int32)

    def get_integer_lower_bounds(self):
        return np.array([self._lower_bounds, ], dtype=np.int32)

    def get_integer_upper_bounds(self):
        return np.array([self._upper_bounds, ], dtype=np.int32)

    def index(self, timestep, scenario_index):
        """Returns the current index"""
        # return index as an integer
        return self._value

    @classmethod
    def load(cls, model, data):
        if "value" in data:
            value = data.pop("value")
        else:
            value = load_parameter_values(model, data)
        parameter = cls(model, value, **data)
        return parameter


IndexVariableParameter.register()


class IrrigationWaterRequirementParameter(Parameter):
    """Simple irrigation water requirement model. """
    def __init__(self, model, rainfall_parameter, et_parameter, crop_water_factor_parameter, area, reference_et, yield_per_area, conveyance_efficiency, 
                 application_efficiency, factor=1e6, revenue_per_yield=1,  et_factor=0.001, area_factor=10000, **kwargs):

        super().__init__(model, **kwargs)

        self._area = None
        self.area = area
        self.factor = factor
        self.et_factor = et_factor
        self.area_factor = area_factor
        self._et_parameter = None
        self._rainfall_parameter = None
        self.reference_et = reference_et
        self.et_parameter = et_parameter
        self._yield_per_area = None
        self.yield_per_area = yield_per_area
        self._crop_water_factor_parameter = None
        self.revenue_per_yield = revenue_per_yield
        self.rainfall_parameter = rainfall_parameter
        self._conveyance_efficiency = None
        self.conveyance_efficiency = conveyance_efficiency
        self._application_efficiency = None
        self.application_efficiency = application_efficiency
        self.crop_water_factor_parameter = crop_water_factor_parameter

    
    yield_per_area = parameter_property("_yield_per_area")
    rainfall_parameter = parameter_property("_rainfall_parameter")
    crop_water_factor_parameter = parameter_property("_crop_water_factor_parameter")
    conveyance_efficiency = parameter_property("_conveyance_efficiency")
    application_efficiency = parameter_property("_application_efficiency")
    area = parameter_property("_area")

    def value(self, timestep, scenario_index):

        et = self.et_parameter.get_value(scenario_index) * self.et_factor
        effective_rainfall = self.rainfall_parameter.get_value(scenario_index) * self.et_factor
        crop_water_factor = self.crop_water_factor_parameter.get_value(scenario_index)
        conv_efficiency = self.conveyance_efficiency.get_value(scenario_index)
        app_efficiency = self.application_efficiency.get_value(scenario_index)
        area_ = self.area.get_value(scenario_index)
      
        # Calculate crop water requirement
        if effective_rainfall > crop_water_factor * et:
            # No crop water requirement if there is enough rainfall
            crop_water_requirement = 0.0
        else:
            # Irrigation required to meet shortfall in rainfall
            
            crop_water_requirement = (crop_water_factor * et - effective_rainfall) * (area_ * self.area_factor)

        # Calculate overall efficiency
        efficiency = app_efficiency * conv_efficiency

        # TODO error checking on division by zero
        irrigation_water_requirement = crop_water_requirement / efficiency
        
        return irrigation_water_requirement/self.factor #To have Mm3/day

    def crop_yield(self, curtailment_ratio):
        return self.area * self.yield_per_area * curtailment_ratio

    def crop_revenue(self, curtailment_ratio):
        return self.revenue_per_yield * self.crop_yield(curtailment_ratio)

    @classmethod
    def load(cls, model, data):

        rainfall_parameter = load_parameter(model, data.pop('rainfall_parameter'))
        et_parameter = load_parameter(model, data.pop('et_parameter'))
        cwf_parameter = load_parameter(model, data.pop('crop_water_factor_parameter'))

        attribute_list = ["conveyance_efficiency", "application_efficiency", "area", "reference_et", "yield_per_area"]
        attributes = {}

        for attribute in attribute_list:
            if attribute in data:
                if isinstance(data[attribute], (int, float)):
                    attributes[attribute] = data.pop(attribute)
                else:
                    attributes[attribute] = load_parameter(model, data.pop(attribute))

        return cls(model, rainfall_parameter, et_parameter, cwf_parameter, attributes["area"], attributes["reference_et"], 
                   attributes["yield_per_area"], attributes["conveyance_efficiency"], attributes["application_efficiency"], **data)


IrrigationWaterRequirementParameter.register()


class TransientDecisionParameter(Parameter):
    """ Return one of two values depending on the current time-step

    This `Parameter` can be used to model a discrete decision event
     that happens at a given date. Prior to this date the `before`
     value is returned, and post this date the `after` value is returned.

    Parameters
    ----------
    decision_date : string or pandas.Timestamp
        The trigger date for the decision.
    before_parameter : Parameter
        The value to use before the decision date.
    after_parameter : Parameter
        The value to use after the decision date.
    earliest_date : string or pandas.Timestamp or None
        Earliest date that the variable can be set to. Defaults to `model.timestepper.start`
    latest_date : string or pandas.Timestamp or None
        Latest date that the variable can be set to. Defaults to `model.timestepper.end`
    decision_freq : pandas frequency string (default 'YS')
        The resolution of feasible dates. For example 'YS' would create feasible dates every
        year between `earliest_date` and `latest_date`. The `pandas` functions are used
        internally for delta date calculations.

    """

    def __init__(self, model, decision_date, before_parameter, after_parameter,
                 earliest_date=None, latest_date=None, decision_freq='YS', **kwargs):
        super(TransientDecisionParameter, self).__init__(model, **kwargs)
        self._decision_date = None
        self.decision_date = decision_date

        if not isinstance(before_parameter, Parameter):
            raise ValueError('The `before` value should be a Parameter instance.')
        before_parameter.parents.add(self)
        self.before_parameter = before_parameter

        if not isinstance(after_parameter, Parameter):
            raise ValueError('The `after` value should be a Parameter instance.')
        after_parameter.parents.add(self)
        self.after_parameter = after_parameter

        # These parameters are mostly used if this class is used as variable.
        self._earliest_date = None
        self.earliest_date = earliest_date

        self._latest_date = None
        self.latest_date = latest_date

        self.decision_freq = self._normalise_decision_freq(decision_freq)
        self._feasible_dates = None
        self.integer_size = 1  # This parameter has a single integer variable

    @staticmethod
    def _normalise_decision_freq(freq):
        if isinstance(freq, str) and (freq == 'AS' or freq.startswith('AS-')):
            return 'YS' + freq[2:]
        return freq

    def decision_date():
        def fget(self):
            return self._decision_date
        
        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._decision_date = value
            else:
                self._decision_date = pd.to_datetime(value)

        return locals()

    decision_date = property(**decision_date())

    def earliest_date():
        def fget(self):
            if self._earliest_date is not None:
                return self._earliest_date
            else:
                return self.model.timestepper.start

        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._earliest_date = value
            else:
                self._earliest_date = pd.to_datetime(value)

        return locals()

    earliest_date = property(**earliest_date())

    def latest_date():
        def fget(self):
            if self._latest_date is not None:
                return self._latest_date
            else:
                return self.model.timestepper.end

        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._latest_date = value
            else:
                self._latest_date = pd.to_datetime(value)

        return locals()

    latest_date = property(**latest_date())

    def setup(self):
        super(TransientDecisionParameter, self).setup()

        # Now setup the feasible dates for when this object is used as a variable.
        self._feasible_dates = pd.date_range(self.earliest_date, self.latest_date,
                                                 freq=self.decision_freq)
        
    def value(self, ts, scenario_index):

        if ts is None:
            v = self.before_parameter.get_value(scenario_index)
        elif ts.datetime >= self.decision_date:
            v = self.after_parameter.get_value(scenario_index)
        else:
            v = self.before_parameter.get_value(scenario_index)
        return v

    def get_integer_lower_bounds(self):
        return np.array([0, ], dtype=np.int)

    def get_integer_upper_bounds(self):
        return np.array([len(self._feasible_dates) - 1, ], dtype=np.int)

    def set_integer_variables(self, values):
        # Update the decision date with the corresponding feasible date
        self.decision_date = self._feasible_dates[values[0]]

    def get_integer_variables(self):
        return np.array([self._feasible_dates.get_loc(self.decision_date), ], dtype=np.int)

    def dump(self):

        data = {
            'earliest_date': self.earliest_date.isoformat(),
            'latest_date': self.latest_date.isoformat(),
            'decision_date': self.decision_date.isoformat(),
            'decision_frequency': self.decision_freq
        }

        return data

    @classmethod
    def load(cls, model, data):

        before_parameter = load_parameter(model, data.pop('before_parameter'))
        after_parameter = load_parameter(model, data.pop('after_parameter'))

        return cls(model, before_parameter=before_parameter, after_parameter=after_parameter, **data)
        
        
TransientDecisionParameter.register()


class RollingNodeFlowRecorder(NodeRecorder):
    """ Records the mean flow of a node for the previous N timesteps (a window).

    This recorder is different to `RollingMeanFlowNodeRecorder` because for the timesteps lower that the window
    we save a value passed in the recorder definition

    Parameters
    ----------

    mode : `pywr.core.Model`
    node : `pywr.core.Node`
        The node to record
    window : int
        The number of timesteps to calculate the flow rolling mean
    hist_rolling : int
        The value to be saved when the timestep is lower that the window
    name : str (optional)
        The name of the recorder

    """

    def __init__(self, model, node, window=None, days=None, hist_rolling=None, name=None, **kwargs):
        super(RollingNodeFlowRecorder, self).__init__(model, node, name=name, **kwargs)
        # self.model = model

        if not window and not days:
            raise ValueError("Either `window` or `days` must be specified.")
        if window:
            self.window = int(window)
        else:
            self.window = 0
        if days:
            self.days = int(days)
        else:
            self.days = 0

        self._data = None
        self.position = 0

        if not hist_rolling:
            raise ValueError("An `hist_rolling` must be specified.")
        else:
            self.hist_rolling = hist_rolling

    def setup(self):
        super(RollingNodeFlowRecorder, self).setup()
        self._data = np.empty([len(self.model.timestepper), len(self.model.scenarios.combinations)])

        if self.days > 0:
            try:
                self.window = self.days // self.model.timestepper.delta
            except TypeError:
                raise TypeError('A rolling window defined as a number of days is only valid with daily time-steps.')
        if self.window == 0:
            raise ValueError("window property of MeanFlowRecorder is less than 1.")

        self._memory = np.zeros([len(self.model.scenarios.combinations), self.window])

    def reset(self):
        super(RollingNodeFlowRecorder, self).reset()
        self.position = 0
        self._memory[:, :] = 0
        self._data[:, :] = 0.0

    def after(self):

        # Save today's flow
        for i in range(0, self._memory.shape[0]):
            self._memory[i, self.position] = self.node.flow[i]

        # Calculate the mean flow
        timestep = self.model.timestepper.current
        if timestep.index < self.window:
            n = timestep.index + 1
        else:
            n = self.window

        # Save the mean flow
        if timestep.index < self.window:
            self._data[int(timestep.index), :] = self.hist_rolling
        else:
            mean_flow = np.mean(self._memory[:, 0:n], axis=1)
            self._data[int(timestep.index), :] = mean_flow

        # Prepare for the next timestep
        self.position += 1
        if self.position >= self.window:
            self.position = 0

    @property
    def data(self):
        return np.array(self._data, dtype=np.float64)

    def to_dataframe(self):
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        return pd.DataFrame(data=self.data, index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        name = data.get("name")
        # node = model.nodes[data["node"]] # for the new version of pywr
        node = model._get_node_from_ref(model, data.pop("node"))

        if "hist_rolling" in data:
            hist_rolling = data["hist_rolling"]
        else:
            hist_rolling = None

        if "window" in data:
            window = int(data["window"])
        else:
            window = None

        if "days" in data:
            days = int(data["days"])
        else:
            days = None

        return cls(model, node, window=window, days=days, hist_rolling=hist_rolling, name=name)


RollingNodeFlowRecorder.register()


#class IndexedArrayParameter(Parameter):
#    """Parameter which uses an IndexParameter to index an array of Parameters
#    An example use of this parameter is to return a demand saving factor (as
#    a float) based on the current demand saving level (calculated by an
#    `IndexParameter`).
#    Parameters
#    ----------
#    index_parameter : `IndexParameter`
#    params : iterable of `Parameters` or floats
#    Notes
#    -----
#    Float arguments `params` are converted to `ConstantParameter`
#    """

#    def __init__(self, model, index_parameter, params, **kwargs):
#        super().__init__(model, **kwargs)
#        assert(isinstance(index_parameter, IndexParameter))
#        self.index_parameter = index_parameter
#        self.children.add(index_parameter)

#        self.params = []
#        for p in params:
#            if not isinstance(p, Parameter):
#                p = ConstantParameter(model, p)
#                from pywr.parameters import ConstantParameter
#            self.params.append(p)

#        for param in self.params:
#            self.children.add(param)
#        self.children.add(index_parameter)

#    def value(self, timestep, scenario_index):
#        """Returns the value of the Parameter at the current index"""
#        #index = self.index_parameter.get_index(scenario_index)
        
#        index = self.index_parameter.get_integer_variables()[0]
#        parameter = self.params[index]
#        return parameter.get_value(scenario_index)

#    @classmethod
#    def load(cls, model, data):
#        index_parameter = load_parameter(model, data.pop("index_parameter"))
#        try:
#            parameters = data.pop("params")
#        except KeyError:
#            parameters = data.pop("parameters")
#        parameters = [load_parameter(model, parameter_data) for parameter_data in parameters]
#        return cls(model, index_parameter, parameters, **data)

        
#IndexedArrayParameter.register()

# ==============================================================================
# From here some parameters created by Mikiyas for the Incomati Basin model
# ==============================================================================
class Domestic_deamnd_projection_parameter(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, Annual_increase_in_percent, **kwargs):
        super().__init__(model, **kwargs)
        self.Annual_increase_in_percent = Annual_increase_in_percent
        
    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        
    def value(self, timestep, scenario_index):    
        
        ts_start_year = self.model.timestepper.start.year
        ts_year=self.model.timestepper.current.year
        year_diff = ts_year - ts_start_year
        projected_demand_factor = self.Annual_increase_in_percent**year_diff
        return projected_demand_factor
        
    @classmethod
    def load(cls, model, data):
        Annual_increase_in_percent = data.pop("Annual_increase_in_percent")
    
        return cls(model, Annual_increase_in_percent, **data)
    
Domestic_deamnd_projection_parameter.register()


class Demand_informed_release_Driekoppies(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, demand_nodes, DS_Komati_Lomati, Maguga, Driekoppies, Buffer_volume_Driekoppies, Buffer_volume_Maguga, Demand_storage_Maguga, Demand_storage_Driekoppies, **kwargs):
        super().__init__(model, **kwargs)
        self.demand_nodes = demand_nodes
        self.DS_Komati_Lomati = DS_Komati_Lomati        
        
        self.Demand_storage_Maguga = Demand_storage_Maguga
        self.Demand_storage_Driekoppies = Demand_storage_Driekoppies 
        self.Maguga =  Maguga
        self.Driekoppies = Driekoppies
        self.Buffer_volume_Driekoppies = Buffer_volume_Driekoppies 
        self.Buffer_volume_Maguga = Buffer_volume_Maguga 

    def setup(self):
        super().setup()

    def value(self, ts, scenario_index):
        
        ts = self.model.timestepper.current
        Buffer_volume_Driekoppies = self.Buffer_volume_Driekoppies.value(ts, scenario_index) 
        Buffer_volume_Maguga = self.Buffer_volume_Maguga.value(ts, scenario_index) 
    
        Maguga_volume = self.Maguga.volume[scenario_index.global_id]
        Driekoppies_volume = self.Driekoppies.volume[scenario_index.global_id]

        if ts.index == 0:
            self.Irrigation_us_Driekoppies = ["X14_RR1","X14_RR2","X14_RR3", "X14_RR5","X14_RR6","X14_RR8"]
            Irrigation_ds_Driekoppies = ["X14_RR5","X14_RR6","X14_RR8"]
            self.Irrigation_us_Driekoppies_max_flow = {node:load_parameter(self.model, node+".Demand") for node in self.Irrigation_us_Driekoppies} 


        #if self.Driekoppies.volume[scenario_index.global_id] < Buffer_volume_Driekoppies:
        #    for node in self.Irrigation_us_Driekoppies:
        #        self.model._get_node_from_ref(self.model, node).max_flow = self.Irrigation_us_Driekoppies_max_flow[node].get_value(scenario_index)*0.5
        #else:
        #    for node in self.Irrigation_us_Driekoppies:
        #        self.model._get_node_from_ref(self.model, node).max_flow = self.Irrigation_us_Driekoppies_max_flow[node].get_value(scenario_index)

        demand = [node.get_max_flow(scenario_index) for node in self.demand_nodes]
        DS_Komati_Lomati = [node.get_max_flow(scenario_index) for node in self.DS_Komati_Lomati]

        #if both Maguga and Driekoppies volume are above the buffer storage volume 
        if (Maguga_volume > Buffer_volume_Maguga and Driekoppies_volume > Buffer_volume_Driekoppies): 
            net_storage_Maguga = Maguga_volume - Buffer_volume_Maguga 
            net_storage_Driekoppies = Driekoppies_volume - Buffer_volume_Driekoppies 
            Driekoppies_precent_relese = (net_storage_Driekoppies)/(net_storage_Maguga + net_storage_Driekoppies)
            Driekoppies_precent_relese = 0 if np.isnan(Driekoppies_precent_relese) else Driekoppies_precent_relese
            release = sum(demand) + sum(DS_Komati_Lomati)*Driekoppies_precent_relese

        elif (Maguga_volume < Buffer_volume_Maguga and Driekoppies_volume < Buffer_volume_Driekoppies):
            net_storage_Maguga = Maguga_volume - self.Demand_storage_Maguga 
            net_storage_Driekoppies = Driekoppies_volume - self.Demand_storage_Driekoppies 
            Driekoppies_precent_relese = (net_storage_Driekoppies)/(net_storage_Maguga + net_storage_Driekoppies)
            Driekoppies_precent_relese = 0 if np.isnan(Driekoppies_precent_relese) else Driekoppies_precent_relese
            release = sum(demand) + sum(DS_Komati_Lomati)*Driekoppies_precent_relese

        elif (Maguga_volume < Buffer_volume_Maguga and Driekoppies_volume > Buffer_volume_Driekoppies):
            release = sum(demand) + sum(DS_Komati_Lomati)

        elif (Maguga_volume > Buffer_volume_Maguga and Driekoppies_volume < Buffer_volume_Driekoppies):
            release = sum(demand) 

        return release

    @classmethod
    def load(cls, model, data):
        demand_nodes = [model._get_node_from_ref(model, node) for node in data.pop("demand_nodes")]
        DS_Komati_Lomati = [model._get_node_from_ref(model, node) for node in data.pop("DS_Komati_Lomati")]

        Maguga =  model._get_node_from_ref(model, data.pop("Maguga_Dam"))
        Driekoppies = model._get_node_from_ref(model, data.pop("Driekoppies_Dam"))
        Demand_storage_Maguga = data.pop("Demand_storage_Maguga")
        Demand_storage_Driekoppies = data.pop("Demand_storage_Driekoppies")

        Buffer_volume_Driekoppies = load_parameter(model, data.pop("Buffer_volume_Driekoppies"))
        Buffer_volume_Maguga = load_parameter(model, data.pop("Buffer_volume_Maguga"))

        return cls(model, demand_nodes, DS_Komati_Lomati, Maguga, Driekoppies, Buffer_volume_Driekoppies, Buffer_volume_Maguga, Demand_storage_Maguga, Demand_storage_Driekoppies, **data)

Demand_informed_release_Driekoppies.register()


class High_assurance_level(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, High_assurance_nodes, **kwargs):
        super().__init__(model, **kwargs)
        self.High_assurance_names = High_assurance_nodes 
        self.High_assurance_nodes = [model._get_node_from_ref(model, node) for node in self.High_assurance_names]
        
    def setup(self):
        super().setup()
        self.demand = []
        
    def probability_exceedance_plot(self, values):
        # Sort the values in ascending order
        sorted_values = np.sort(values)
        # Calculate the exceedance probabilities
        exceedance_probabilities = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        exceedance_probabilities = 1 - exceedance_probabilities
                # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_values, exceedance_probabilities, marker='o', linestyle='-')
        plt.xlabel('Value')
        plt.ylabel('Exceedance Probability')
        plt.title('Probability Exceedance Plot')
        plt.grid(True)
        plt.show()

    def after(self):
        self.max_demand = {node:load_parameter(self.model, node+".Demand") for node in self.High_assurance_names}  
        agg_demand = []
        self.max_demand = {node:load_parameter(self.model, node+".Demand") for node in self.High_assurance_names} 
        for node in self.High_assurance_names:
            allocated_demand = self.model._get_node_from_ref(self.model, node)
            potential_demand = self.max_demand[node].get_value(self.scenario_id) 
            
            agg_demand.append(potential_demand - allocated_demand.flow[self.scenario_id.global_id])

        self.demand.append(sum(agg_demand))

        if not self.ts.year % 10 and self.ts.month == 12 and self.ts.day == 31:
            #self.probability_exceedance_plot(self.demand)
            pass

        return 0

    def value(self, ts, scenario_index):
        
        ts = self.model.timestepper.current
        self.ts = ts
        self.scenario_id = scenario_index

        agg_demand = []
        self.max_demand = {node:load_parameter(self.model, node+".Demand") for node in self.High_assurance_names} 
        for node in self.High_assurance_names:
            allocated_demand = self.model._get_node_from_ref(self.model, node)
            potential_demand = self.max_demand[node].get_value(scenario_index) 
            
            agg_demand.append(potential_demand - allocated_demand.prev_flow[scenario_index.global_id])

        """
        self.demand.append(sum(agg_demand))

        if ts.year % 3 and ts.month == 12:
            self.probability_exceedance_plot(self.demand)
        """
        return 0

    @classmethod
    def load(cls, model, data):
        High_assurance_nodes = data.pop("High_assurance_nodes")
        return cls(model, High_assurance_nodes, **data)

High_assurance_level.register()


class Low_assurance_level(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, High_assurance_nodes, **kwargs):
        super().__init__(model, **kwargs)
        self.High_assurance_names = High_assurance_nodes 
        self.High_assurance_nodes = {node: model._get_node_from_ref(model, node) for node in self.High_assurance_names}
        
    def setup(self):
        super().setup()
        self.allocated_demand = 0
        self.potential_demand = {node: 0 for node in self.High_assurance_names}
        self.demand = []
        self.max_demand = {node:load_parameter(self.model, node+".Demand") for node in self.High_assurance_names} 

    def probability_exceedance_plot(self, values):
        # Sort the values in ascending order
        sorted_values = np.sort(values)
        # Calculate the exceedance probabilities
        exceedance_probabilities = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        exceedance_probabilities = 1 - exceedance_probabilities
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_values, exceedance_probabilities, marker='o', linestyle='-')
        plt.xlabel('Value')
        plt.ylabel('Exceedance Probability')
        plt.title('Probability Exceedance Plot')
        plt.grid(True)
        plt.show()

    def value(self, ts, scenario_index):
        
        ts = self.model.timestepper.current
        self.ts = ts
        self.scenario_id = scenario_index

        agg_demand = []

        for node in self.High_assurance_names:
            self.allocated_demand = self.model._get_node_from_ref(self.model, node)
            agg_demand.append(self.potential_demand[node] - self.allocated_demand.prev_flow[scenario_index.global_id])
            self.potential_demand[node] = self.max_demand[node].get_value(scenario_index) 

        if not ts.index == 0:
            self.demand.append(sum(agg_demand))


        if not self.ts.year % 5 and self.ts.month == 12 and self.ts.day == 31:
            #self.probability_exceedance_plot(self.demand)
            pass
            
        return 0

    @classmethod
    def load(cls, model, data):
        High_assurance_nodes = data.pop("High_assurance_nodes")
        return cls(model, High_assurance_nodes, **data)

Low_assurance_level.register()


class Simple_Irr_demand_calculator_with_file(Parameter):
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
                Irr_eff=1+(1-0.80)
                Irr_demand=Net_demand*(self.crop_area[x]*0.01*1.05)*self.Max_area*Irr_eff * 1e6 * 1e-3 * 1e-6
                
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

Simple_Irr_demand_calculator_with_file.register()


class Demand_informed_release_Maguga(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, demand_nodes, DS_Komati_Lomati, Maguga, Driekoppies, Buffer_volume_Driekoppies, Buffer_volume_Maguga, Demand_storage_Maguga, Demand_storage_Driekoppies, Maguga_hydropower_release_m3_s, Maguga_hydropower_operation_hours_per_week, **kwargs):
        super().__init__(model, **kwargs)
        self.demand_nodes = demand_nodes
        self.DS_Komati_Lomati = DS_Komati_Lomati
        self.Demand_storage_Maguga = Demand_storage_Maguga
        self.Demand_storage_Driekoppies = Demand_storage_Driekoppies 
        self.Maguga =  Maguga
        self.Driekoppies = Driekoppies
        self.Buffer_volume_Driekoppies = Buffer_volume_Driekoppies 
        self.Buffer_volume_Maguga = Buffer_volume_Maguga 

        self.Maguga_hydropower_release_m3_s = Maguga_hydropower_release_m3_s
        self.Maguga_hydropower_operation_hours_per_week = Maguga_hydropower_operation_hours_per_week 

    def setup(self):
        super().setup()
        self.Irrigation_us_Maguga = ["X11_RR3","X11_RR4","X11_RR16","X11_RR7","X11_RR9","X11_RR11","X11_RR13","X12_RR12","X12_RR13","X12_RR9","X12_RR7","X13_RR1","X13_RR3","X13_RR5","X13_RR7","X13_RR12","X13_RR2"]
        self.Irrigation_us_Maguga_max_flow = {node:load_parameter(self.model, node+".Demand") for node in self.Irrigation_us_Maguga} 
        self.peak_hours_relese = self.Maguga_hydropower_release_m3_s["peak_hours_relese"] 
        self.standard_hours_generation = self.Maguga_hydropower_release_m3_s["standard_hours_generation"] 
        self.off_peak_hours_relese = self.Maguga_hydropower_release_m3_s["off_peak_hours_relese"] 

        self.peak_hours = load_parameter(self.model, self.Maguga_hydropower_operation_hours_per_week["peak_hours"])
        self.standard_hours = load_parameter(self.model, self.Maguga_hydropower_operation_hours_per_week["standard_hours"])
        
    def value(self, ts, scenario_index):
        
        ts = self.model.timestepper.current
        
        self.peak_hour = self.peak_hours.get_value(scenario_index)
        self.standard_hour = self.standard_hours.get_value(scenario_index)
        self.off_peak_hour = 168 - self.peak_hour - self.standard_hour

        factor = 3600/7e6
        self.hydro_release = (self.peak_hour*self.peak_hours_relese + self.standard_hour*self.standard_hours_generation + self.off_peak_hour*self.off_peak_hours_relese)*factor


        Buffer_volume_Driekoppies = self.Buffer_volume_Driekoppies.value(ts, scenario_index) 
        Buffer_volume_Maguga = self.Buffer_volume_Maguga.value(ts, scenario_index) 
        Maguga_volume = self.Maguga.volume[scenario_index.global_id]
        Driekoppies_volume = self.Driekoppies.volume[scenario_index.global_id]


        if self.Maguga.volume[scenario_index.global_id] < Buffer_volume_Maguga:
            for node in self.Irrigation_us_Maguga:
                self.model._get_node_from_ref(self.model, node).max_flow = self.Irrigation_us_Maguga_max_flow[node].get_value(scenario_index)
        else:
            for node in self.Irrigation_us_Maguga:
                self.model._get_node_from_ref(self.model, node).max_flow = self.Irrigation_us_Maguga_max_flow[node].get_value(scenario_index)

        #Todo
        # 1. identfy the hydropwoer requirment as an input parameter. this could be the montly required release based on the Peak, Standard and __
        # 2. compaire the downstream release with the hydropower demand and if the hydropower release is above the demand, adjust the release 
        # 3. make sure that the demand is satisfied unless there is no water in the reservoir.
        # 4. make sure that the release form the reservoir should consider the spill from the reservoir


        demand = [node.get_max_flow(scenario_index) for node in self.demand_nodes]
        DS_Komati_Lomati = [node.get_max_flow(scenario_index) for node in self.DS_Komati_Lomati]
        #if both Maguga and Driekoppies volume are above the buffer storage volume
        Maguga_precent_relese = 0 
        if (Maguga_volume > Buffer_volume_Maguga and Driekoppies_volume > Buffer_volume_Driekoppies): 
            net_storage_maguga = Maguga_volume - Buffer_volume_Maguga 
            net_storage_Driekoppies = Driekoppies_volume - Buffer_volume_Driekoppies 
            Maguga_precent_relese = (net_storage_maguga)/(net_storage_maguga + net_storage_Driekoppies)
            Maguga_precent_relese = 0 if np.isnan(Maguga_precent_relese) else Maguga_precent_relese
            release = sum(demand) + sum(DS_Komati_Lomati)*Maguga_precent_relese

        elif (Maguga_volume < Buffer_volume_Maguga and Driekoppies_volume < Buffer_volume_Driekoppies):
            net_storage_maguga = Maguga_volume - self.Demand_storage_Maguga 
            net_storage_Driekoppies = Driekoppies_volume - self.Demand_storage_Driekoppies 
            Maguga_precent_relese = (net_storage_maguga)/(net_storage_maguga + net_storage_Driekoppies)
            Maguga_precent_relese = 0 if np.isnan(Maguga_precent_relese) else Maguga_precent_relese
            release = sum(demand) + sum(DS_Komati_Lomati)*Maguga_precent_relese

        elif (Maguga_volume > Buffer_volume_Maguga and Driekoppies_volume < Buffer_volume_Driekoppies):
            release = sum(demand) + sum(DS_Komati_Lomati)

        elif (Maguga_volume < Buffer_volume_Maguga and Driekoppies_volume > Buffer_volume_Driekoppies):
            release = sum(demand) 
     
        # release for hydropower  
        if release < self.hydro_release:
            release = self.hydro_release

        return release

    @classmethod
    def load(cls, model, data):
        demand_nodes = [model._get_node_from_ref(model, node) for node in data.pop("demand_nodes")]
        DS_Komati_Lomati = [model._get_node_from_ref(model, node) for node in data.pop("DS_Komati_Lomati")]
        Maguga =  model._get_node_from_ref(model, data.pop("Maguga_Dam"))
        Driekoppies = model._get_node_from_ref(model, data.pop("Driekoppies_Dam"))
        Demand_storage_Maguga = data.pop("Demand_storage_Maguga")
        Demand_storage_Driekoppies = data.pop("Demand_storage_Driekoppies")

        Buffer_volume_Driekoppies = load_parameter(model, data.pop("Buffer_volume_Driekoppies"))
        Buffer_volume_Maguga = load_parameter(model, data.pop("Buffer_volume_Maguga"))

        Maguga_hydropower_release_m3_s = data.pop("Maguga_hydropower_release_m3_s")
        Maguga_hydropower_operation_hours_per_week = data.pop("Maguga_hydropower_operation_hours_per_week")

        return cls(model, demand_nodes, DS_Komati_Lomati, Maguga, Driekoppies, Buffer_volume_Driekoppies, Buffer_volume_Maguga, Demand_storage_Maguga, Demand_storage_Driekoppies, Maguga_hydropower_release_m3_s, Maguga_hydropower_operation_hours_per_week, **data)

Demand_informed_release_Maguga.register()

class SeasonalStorageTargetParameter(Parameter):
    """
    Annual, state-dependent storage target for a reservoir.

    Purpose
    -------
    This parameter computes one target storage value per water year and scenario.
    The target is recalculated on or after a decision date (e.g. 1 April) and then
    held constant until the next water year.

    The target is designed for future-operation optimisation where no observed
    storage target exists. It balances:
      - irrigation pressure / irrigation value  -> lowers target (release more)
      - winter energy pressure / value          -> raises target (store more)
      - drought / low availability              -> raises target (hedging)

    Notes
    -----
    - This class uses *previously realised* inflow (from completed timesteps only)
      plus optional expected remaining inflow / losses parameters.
    - Default units assume storage in Mm3 and flow in Mm3/day if flow_is_per_day=True.
    - If you want a more explicit annual economic signal, pass:
        irrigation_value_parameter
        winter_energy_value_parameter
      Otherwise demand and current storage are used as proxies.

    Suggested Toktogul-like defaults (Mm3):
      lower_target = 10500   # cooperative-ish end-of-vegetation target
      upper_target = 13000   # more conservative / energy-security target
      min_volume   =  6500   # safe operating floor
    """

    def __init__(
        self,
        model,
        storage_node,
        inflow_nodes=None,
        demand_nodes=None,
        expected_remaining_inflow_parameter=None,
        expected_remaining_losses_parameter=None,
        irrigation_value_parameter=None,
        winter_energy_value_parameter=None,
        decision_month=4,
        decision_day=1,
        water_year_start_month=10,
        water_year_start_day=1,
        lower_target=10500.0,
        upper_target=13000.0,
        min_volume=6500.0,
        max_volume=None,
        energy_weight=1.0,
        irrigation_weight=1.0,
        drought_weight=1.0,
        irrigation_norm=1.0,
        energy_norm=1.0,
        availability_norm=None,
        stickiness=0.0,
        flow_is_per_day=True,
        **kwargs,
    ):
        super().__init__(model, **kwargs)

        self.storage_node = storage_node
        self.inflow_nodes = inflow_nodes if inflow_nodes is not None else []
        self.demand_nodes = demand_nodes if demand_nodes is not None else []

        self.expected_remaining_inflow_parameter = expected_remaining_inflow_parameter
        self.expected_remaining_losses_parameter = expected_remaining_losses_parameter
        self.irrigation_value_parameter = irrigation_value_parameter
        self.winter_energy_value_parameter = winter_energy_value_parameter

        self.decision_month = int(decision_month)
        self.decision_day = int(decision_day)
        self.water_year_start_month = int(water_year_start_month)
        self.water_year_start_day = int(water_year_start_day)

        self.lower_target = float(lower_target)
        self.upper_target = float(upper_target)
        self.min_volume = float(min_volume)
        self.max_volume = None if max_volume is None else float(max_volume)

        self.energy_weight = float(energy_weight)
        self.irrigation_weight = float(irrigation_weight)
        self.drought_weight = float(drought_weight)

        self.irrigation_norm = float(irrigation_norm)
        self.energy_norm = float(energy_norm)
        self.availability_norm = availability_norm

        self.stickiness = float(stickiness)
        self.flow_is_per_day = bool(flow_is_per_day)

        # Register dependencies
        for p in [
            self.expected_remaining_inflow_parameter,
            self.expected_remaining_losses_parameter,
            self.irrigation_value_parameter,
            self.winter_energy_value_parameter,
        ]:
            if isinstance(p, Parameter):
                self.children.add(p)

        # Internal state
        self._current_target = None
        self._target_water_year = None
        self._observed_inflow_ytd = None
        self._last_seen_timestep_index = None

    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)

        self._current_target = np.full(ncomb, self.lower_target, dtype=np.float64)
        self._target_water_year = np.full(ncomb, -9999, dtype=np.int32)
        self._observed_inflow_ytd = np.zeros(ncomb, dtype=np.float64)
        self._last_seen_timestep_index = None

    def reset(self):
        super().reset()
        self._current_target[:] = self.lower_target
        self._target_water_year[:] = -9999
        self._observed_inflow_ytd[:] = 0.0
        self._last_seen_timestep_index = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_timestamp(self, value):
        if isinstance(value, pd.Timestamp):
            return value
        if hasattr(value, "to_timestamp"):
            return value.to_timestamp()
        return pd.Timestamp(value)

    def _water_year(self, dt):
        dt = self._to_timestamp(dt)
        wy_start = pd.Timestamp(year=dt.year, month=self.water_year_start_month, day=self.water_year_start_day)
        if dt >= wy_start:
            return dt.year + 1
        return dt.year

    def _decision_datetime_for_water_year(self, water_year):
        # If decision month is in/after the water-year start month, it belongs to calendar year water_year-1
        if self.decision_month >= self.water_year_start_month:
            year = water_year - 1
        else:
            year = water_year
        return pd.Timestamp(year=year, month=self.decision_month, day=self.decision_day)

    def _get_parameter_value(self, obj, scenario_index, default=0.0):
        if obj is None:
            return float(default)
        if isinstance(obj, Parameter):
            return float(obj.get_value(scenario_index))
        return float(obj)

    def _timestep_days_between_indices(self, prev_idx, curr_idx):
        dt_index = self.model.timestepper.datetime_index
        prev_dt = self._to_timestamp(dt_index[prev_idx])
        curr_dt = self._to_timestamp(dt_index[curr_idx])

        days = (curr_dt - prev_dt).days
        if days > 0:
            return float(days)

        # Fallbacks
        delta = getattr(self.model.timestepper, "delta", None)
        if delta is not None:
            try:
                return float(delta)
            except Exception:
                pass

        return 1.0

    def _advance_observed_inflow_ytd(self, ts):
        """
        Advance internal cumulative observed inflow using completed timesteps only.

        At timestep t, this method adds inflow from timestep t-1 (if available).
        This avoids direct dependence on the current timestep's not-yet-realised flow.
        """
        idx = int(ts.index)

        if self._last_seen_timestep_index is None:
            self._last_seen_timestep_index = idx
            return

        if idx == self._last_seen_timestep_index:
            return

        prev_idx = self._last_seen_timestep_index
        dt_index = self.model.timestepper.datetime_index

        prev_dt = self._to_timestamp(dt_index[prev_idx])
        curr_dt = self._to_timestamp(dt_index[idx])

        prev_wy = self._water_year(prev_dt)
        curr_wy = self._water_year(curr_dt)

        # Reset YTD accumulator at the start of a new water year
        if curr_wy != prev_wy:
            self._observed_inflow_ytd[:] = 0.0
        else:
            dt_days = self._timestep_days_between_indices(prev_idx, idx)

            for scenario_index in self.model.scenarios.combinations:
                gid = scenario_index.global_id
                inflow = 0.0
                for node in self.inflow_nodes:
                    inflow += node.flow[gid]

                if self.flow_is_per_day:
                    inflow *= dt_days

                self._observed_inflow_ytd[gid] += inflow

        self._last_seen_timestep_index = idx

    def _compute_irrigation_signal(self, scenario_index):
        """
        Irrigation pressure / value signal.

        If irrigation_value_parameter is supplied, use it directly.
        Otherwise use the sum of current max_flow across demand nodes as a proxy.
        """
        if self.irrigation_value_parameter is not None:
            return max(self._get_parameter_value(self.irrigation_value_parameter, scenario_index, 0.0), 0.0)

        if not self.demand_nodes:
            return 0.0

        demand = 0.0
        for node in self.demand_nodes:
            demand += node.get_max_flow(scenario_index)
        return max(float(demand), 0.0)

    def _compute_winter_energy_signal(self, scenario_index):
        """
        Winter energy pressure / value signal.

        This should ideally be a projected winter energy value, deficit,
        or revenue proxy passed as a Parameter.
        """
        return max(self._get_parameter_value(self.winter_energy_value_parameter, scenario_index, 0.0), 0.0)

    def _compute_target(self, ts, scenario_index):
        gid = scenario_index.global_id

        storage = float(self.storage_node.volume[gid])

        if self.max_volume is None:
            try:
                max_volume = float(self.storage_node.get_max_volume(scenario_index))
            except Exception:
                max_volume = np.inf
        else:
            max_volume = self.max_volume

        expected_remaining_inflow = self._get_parameter_value(
            self.expected_remaining_inflow_parameter, scenario_index, 0.0
        )
        expected_remaining_losses = self._get_parameter_value(
            self.expected_remaining_losses_parameter, scenario_index, 0.0
        )

        # Forecast available water remaining in the water year.
        available = storage + self._observed_inflow_ytd[gid] + expected_remaining_inflow - expected_remaining_losses
        available = max(available, self.min_volume)

        irrigation_signal = self._compute_irrigation_signal(scenario_index)
        winter_energy_signal = self._compute_winter_energy_signal(scenario_index)

        # Normalised pressures
        ag_score = self.irrigation_weight * (irrigation_signal / max(self.irrigation_norm, 1e-12))
        energy_score = self.energy_weight * (winter_energy_signal / max(self.energy_norm, 1e-12))

        availability_norm = max_volume if self.availability_norm is None else float(self.availability_norm)
        drought_ratio = 1.0 - np.clip(
            (available - self.min_volume) / max(availability_norm - self.min_volume, 1e-12),
            0.0,
            1.0,
        )
        drought_score = self.drought_weight * drought_ratio

        # alpha -> 0 means irrigation-oriented; alpha -> 1 means energy/drought-oriented
        denom = ag_score + energy_score + drought_score
        if denom <= 0.0:
            alpha = 0.5
        else:
            alpha = (energy_score + drought_score) / denom

        target = self.lower_target + alpha * (self.upper_target - self.lower_target)

        # Optional inter-annual smoothing
        if self.stickiness > 0.0 and self._target_water_year[gid] >= 0:
            target = self.stickiness * self._current_target[gid] + (1.0 - self.stickiness) * target

        # Physical bounds
        target = max(target, self.min_volume)
        target = min(target, max_volume)
        target = min(target, available)

        return float(target)

    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current if timestep is None else timestep

        # Update realised inflow from completed timesteps only
        self._advance_observed_inflow_ytd(ts)

        gid = scenario_index.global_id
        dt_index = self.model.timestepper.datetime_index
        dt = self._to_timestamp(dt_index[int(ts.index)])

        wy = self._water_year(dt)
        decision_dt = self._decision_datetime_for_water_year(wy)

        # Compute once per water year, on/after the decision date
        if self._target_water_year[gid] != wy and dt >= decision_dt:
            self._current_target[gid] = self._compute_target(ts, scenario_index)
            self._target_water_year[gid] = wy

        # Before the decision date, return current storage as a neutral target
        if self._target_water_year[gid] != wy:
            return float(self.storage_node.volume[gid])

        return float(self._current_target[gid])

    @classmethod
    def load(cls, model, data):
        storage_node = model._get_node_from_ref(model, data.pop("storage_node"))

        inflow_nodes = [
            model._get_node_from_ref(model, n)
            for n in data.pop("inflow_nodes", [])
        ]

        demand_nodes = [
            model._get_node_from_ref(model, n)
            for n in data.pop("demand_nodes", [])
        ]

        def load_float_or_param(key, default=None):
            if key not in data:
                return default
            value = data.pop(key)
            if isinstance(value, (int, float)):
                return float(value)
            return load_parameter(model, value)

        return cls(
            model,
            storage_node=storage_node,
            inflow_nodes=inflow_nodes,
            demand_nodes=demand_nodes,
            expected_remaining_inflow_parameter=load_float_or_param("expected_remaining_inflow_parameter", None),
            expected_remaining_losses_parameter=load_float_or_param("expected_remaining_losses_parameter", None),
            irrigation_value_parameter=load_float_or_param("irrigation_value_parameter", None),
            winter_energy_value_parameter=load_float_or_param("winter_energy_value_parameter", None),
            decision_month=data.pop("decision_month", 4),
            decision_day=data.pop("decision_day", 1),
            water_year_start_month=data.pop("water_year_start_month", 10),
            water_year_start_day=data.pop("water_year_start_day", 1),
            lower_target=data.pop("lower_target", 10500.0),
            upper_target=data.pop("upper_target", 12500.0),
            min_volume=data.pop("min_volume", 6000.0),
            max_volume=data.pop("max_volume", None),
            energy_weight=data.pop("energy_weight", 1.0),
            irrigation_weight=data.pop("irrigation_weight", 1.0),
            drought_weight=data.pop("drought_weight", 1.0),
            irrigation_norm=data.pop("irrigation_norm", 1.0),
            energy_norm=data.pop("energy_norm", 1.0),
            availability_norm=data.pop("availability_norm", None),
            stickiness=data.pop("stickiness", 0.0),
            flow_is_per_day=data.pop("flow_is_per_day", True),
            **data,
        )


SeasonalStorageTargetParameter.register()
