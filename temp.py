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
            'state_to_args': {
                'args': ['query'],
                'kwargs': {}
            }
        },
        {
            'component': 'OtherComponent',
            'method': 'process',
            'state_to_args': {
                'args': [],
                'kwargs': {'additional_info': 'info'}
            }
        }
    ]
}