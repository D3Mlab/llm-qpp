from utils.setup_logging import setup_logging
from .base_agent import BaseAgent
import types
import copy

class GeneralAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        from .registry import POLICY_CLASSES
        self.policy_class = POLICY_CLASSES.get(self.agent_config.get('policy'))


    def rank(self, query):
        
        #initialize policy
        self.policy = self.policy_class(self.config)

        #state_hist[t] is state at step t
        self.state_hist = [{
            'query' : query
            }]

        #to do - update n_actions_max with a timer/timeout
        n_actions = 0
        n_actions_max = 10
        while n_actions < n_actions_max:

            next_action = self.policy.next_action(self.state_hist[-1])
            #returns act_method or None if no next action
            self.logger.debug(f"next action: {next_action}")

            if not next_action:
                #put state history into result format (dict with current state plus a state history element) and return
                self.logger.debug(f"Next action is None. Returning with state history {self.state_hist} ")
                result = copy.deepcopy(self.state_hist[-1])
                result.update({'state_history' : self.state_hist})
                return result

            act_method = next_action

            #action returns dictionary with some updated/new state variables (e.g. ranked list changes)
            act_result = act_method(self.state_hist[-1])
            #replace any updated state values in the previous state
            curr_state = {**self.state_hist[-1], **act_result}
            curr_state.update({'last_action_method' : act_method.__name__})
            self.logger.debug(curr_state)
            self.state_hist.append(curr_state)

            n_actions += 1
        
        self.logger.warning(f'max number of actions {n_actions_max} reached')
        return None


        