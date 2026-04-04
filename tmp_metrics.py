import json
from pathlib import Path
import torch

from dataset import load_metadata, load_split_csv, create_dataloaders, create_transformer_dataloaders
from evaluate import collect_logits_and_labels, evaluate_from_logits
from models import BiLSTMMultiHeadAttention, TransformerMultiLabelClassifier

root = Path('.')
meta = load_metadata(root / 'preprocessed/metadata.json')
label_order = meta['label_order']
val_df = load_split_csv(root / 'preprocessed/val_preprocessed.csv')

def build_loader_and_model(ckpt_path: Path, batch_size: int = 128):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_type = ckpt.get('model_type', 'bilstm')
    args = ckpt['model_args']
    text_column = ckpt.get('text_column', 'tweet_clean')
    max_length = int(ckpt['max_length'])

    if model_type == 'transformer':
        _, val_loader = create_transformer_dataloaders(
            train_df=val_df,
            val_df=val_df,
            label_order=label_order,
            text_column=text_column,
            tokenizer_name=args['pretrained_model_name'],
            max_length=max_length,
            batch_size=batch_size,
            num_workers=0,
        )
        model = TransformerMultiLabelClassifier(
            pretrained_model_name=args['pretrained_model_name'],
            num_labels=args['num_labels'],
            dropout=args['dropout'],
            multi_sample_dropout=args['multi_sample_dropout'],
            head_type=args['head_type'],
            pooling=args['pooling'],
            freeze_backbone=args.get('freeze_backbone', False),
        )
    else:
        vocab = ckpt['vocab']
        _, val_loader = create_dataloaders(
            train_df=val_df,
            val_df=val_df,
            label_order=label_order,
            text_column=text_column,
            vocab=vocab,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=0,
        )
        model = BiLSTMMultiHeadAttention(
            vocab_size=args['vocab_size'],
            embedding_dim=args['embedding_dim'],
            hidden_size=args['hidden_size'],
            num_layers=args['num_layers'],
            attention_heads=args['attention_heads'],
            dropout=args['dropout'],
            num_labels=args['num_labels'],
        )

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, val_loader

def evaluate_run(name: str, ckpt_path: Path, threshold_path: Path | None):
    model, val_loader = build_loader_and_model(ckpt_path)
    logits, labels = collect_logits_and_labels(model=model, data_loader=val_loader, device=torch.device('cpu'))

    default = evaluate_from_logits(
        logits=logits,
        labels=labels,
        label_order=label_order,
        thresholds=[0.5] * len(label_order),
    )

    print(f'=== {name} ===')
    print('default_macro_f1', round(default['macro_f1'], 6))
    print('default_micro_f1', round(default['micro_f1'], 6))

    if threshold_path is not None:
        payload = json.loads(threshold_path.read_text(encoding='utf-8'))
        thresholds = [float(payload['thresholds'][label]) for label in label_order]
        tuned = evaluate_from_logits(
            logits=logits,
            labels=labels,
            label_order=label_order,
            thresholds=thresholds,
        )
        print('tuned_macro_f1', round(tuned['macro_f1'], 6))
        print('tuned_micro_f1', round(tuned['micro_f1'], 6))
        for label in ['country', 'ingredients', 'conspiracy', 'religious', 'side-effect']:
            i = label_order.index(label)
            print(
                label,
                'default',
                round(default['per_label_f1'][label], 4),
                'tuned',
                round(tuned['per_label_f1'][label], 4),
                'th',
                round(thresholds[i], 2),
            )
    else:
        for label in ['country', 'ingredients', 'conspiracy', 'religious', 'side-effect']:
            print(label, 'default', round(default['per_label_f1'][label], 4))
    print()

if __name__ == '__main__':
    evaluate_run(
        'formal_bilstm',
        root / 'artifacts/formal_bilstm/best_model.pt',
        root / 'artifacts/formal_bilstm/thresholds.json',
    )
    evaluate_run(
        'ctbert_final_no_overfit_v3',
        root / 'artifacts/ctbert_final_no_overfit_v3/best_model.pt',
        root / 'artifacts/ctbert_final_no_overfit_v3/thresholds.json',
    )
    evaluate_run(
        'ctbert_round4_linear_fine',
        root / 'artifacts/ctbert_round4_linear_fine/best_model.pt',
        None,
    )
    evaluate_run(
        'ctbert_round3_label_attn',
        root / 'artifacts/ctbert_round3_label_attn/best_model.pt',
        None,
    )
