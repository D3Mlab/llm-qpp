                # Ensure embedding is a tensor
                if not isinstance(doc['embedding'], torch.Tensor):
                    embedding = torch.tensor(doc['embedding'], dtype=torch.float32)
                else:
                    embedding = doc['embedding']

                # Clone and detach for safety, move to correct dtype
                batch_embeddings.append(embedding.clone().detach().to(dtype=torch.float32)m_scores}