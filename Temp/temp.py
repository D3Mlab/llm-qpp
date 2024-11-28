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
            'component': 'OtherComponent',
            'method': 'process',
        }
    ]
}