from utils.setup_logging import setup_logging
from .base_agent import BaseAgent
import types
#from . import COMPONENT_CLASSES, MAIN_ACTIONS

class GeneralAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        from . import POLICY_CLASSES
        policy_class = POLICY_CLASSES.get(self.agent_config.get('policy'))
        self.policy = policy_class(self.config)

    def rank(self, query):
        
        #state_hist[t] is state at step t
        self.state_hist = [{
            'query' : query
            }]

        n_actions = 0
        n_actions_max = 10
        while n_actions < n_actions_max:
            #to do - update with a timer/timeout
            next_action = self.policy.next_action(self.state_hist[-1])
            #returns (act_method, act_args) or None if no next action

            if not next_action:
                #put state history into result format (dict with current state plus a state history element) and return
                self.logger.debug(f"Next action is None. Returning with state history {self.state_hist} ")
                result = self.state_hist[-1]
                result.update({'state_history' : self.state_hist})
                return result

            act_method, act_args = next_action
            if isinstance(act_args, dict):
                pos_args = act_args.get('args', [])
                kw_args = act_args.get('kwargs', {})

                curr_state = act_method(*pos_args, **kw_args)
                curr_state.update({'last_action_method' : act_method.__name__, 'last_action_args' : act_args})
                self.state_hist.append(curr_state)
            else:
                raise ValueError('Invalid args format')

            n_actions += 1
        
        self.logger.warning(f'max number of actions {n_actions_max} reached')
        return None


        