{
    'agent': {
        'policy': 'PipelinePolicy'
    },
    'components': [
        'DenseRetriever',
        'OtherComponent'
    ],
    'policy_steps': [
        {
            'component': 'DenseRetriever',
            'method': 'rank',
        },
        {
            'component': 'AgentLogic',
            'method': 'check_max_q_reforms',
        }
    ]
}