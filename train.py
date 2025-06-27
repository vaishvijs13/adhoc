import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from models import IdeologyEmbeddingModel, ContrastiveLoss
import argparse
import os
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoliticalTextDataset(Dataset):    
    def __init__(self, texts: List[str], labels: List[int], 
                 social_labels: List[int] = None, 
                 economic_labels: List[int] = None,
                 foreign_labels: List[int] = None):
        self.texts = texts
        self.labels = labels
        self.social_labels = social_labels or [1] * len(texts)
        self.economic_labels = economic_labels or [1] * len(texts)
        self.foreign_labels = foreign_labels or [1] * len(texts)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'political_label': self.labels[idx],
            'social_label': self.social_labels[idx],
            'economic_label': self.economic_labels[idx],
            'foreign_label': self.foreign_labels[idx]
        }

class PoliticalTrainer:    
    def __init__(self, model: IdeologyEmbeddingModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.contrastive_loss = ContrastiveLoss()
        
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, use_contrastive: bool = True):
        self.model.train()
        total_loss = 0
        political_correct = 0
        social_correct = 0
        economic_correct = 0
        foreign_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            texts = batch['text']
            political_labels = batch['political_label'].to(self.device)
            social_labels = batch['social_label'].to(self.device)
            economic_labels = batch['economic_label'].to(self.device)
            foreign_labels = batch['foreign_label'].to(self.device)
            
            results = self.model(texts, return_attention=False)
            
            political_loss = criterion(results['political_logits'], political_labels)
            social_loss = criterion(results['social_logits'], social_labels)
            economic_loss = criterion(results['economic_logits'], economic_labels)
            foreign_loss = criterion(results['foreign_logits'], foreign_labels)
            
            contrastive_loss_val = 0
            if use_contrastive:
                contrastive_loss_val = self.contrastive_loss(
                    results['embeddings'], political_labels
                )
            
            total_loss_val = (political_loss + social_loss + 
                            economic_loss + foreign_loss + 
                            0.1 * contrastive_loss_val)
            
            total_loss_val.backward()
            optimizer.step()
            
            political_pred = torch.argmax(results['political_logits'], dim=1)
            social_pred = torch.argmax(results['social_logits'], dim=1)
            economic_pred = torch.argmax(results['economic_logits'], dim=1)
            foreign_pred = torch.argmax(results['foreign_logits'], dim=1)
            
            political_correct += (political_pred == political_labels).sum().item()
            social_correct += (social_pred == social_labels).sum().item()
            economic_correct += (economic_pred == economic_labels).sum().item()
            foreign_correct += (foreign_pred == foreign_labels).sum().item()
            
            total_samples += len(texts)
            total_loss += total_loss_val.item()
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss_val.item():.4f}',
                'Pol_Acc': f'{political_correct/total_samples:.3f}',
                'Soc_Acc': f'{social_correct/total_samples:.3f}'
            })
        
        return {
            'loss': total_loss / len(dataloader),
            'political_accuracy': political_correct / total_samples,
            'social_accuracy': social_correct / total_samples,
            'economic_accuracy': economic_correct / total_samples,
            'foreign_accuracy': foreign_correct / total_samples
        }
    
    def validate(self, dataloader: DataLoader, criterion: nn.Module):
        self.model.eval()
        total_loss = 0
        political_correct = 0
        social_correct = 0
        economic_correct = 0
        foreign_correct = 0
        total_samples = 0
        
        all_political_preds = []
        all_political_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                texts = batch['text']
                political_labels = batch['political_label'].to(self.device)
                social_labels = batch['social_label'].to(self.device)
                economic_labels = batch['economic_label'].to(self.device)
                foreign_labels = batch['foreign_label'].to(self.device)
                
                results = self.model(texts, return_attention=False)
                
                political_loss = criterion(results['political_logits'], political_labels)
                social_loss = criterion(results['social_logits'], social_labels)
                economic_loss = criterion(results['economic_logits'], economic_labels)
                foreign_loss = criterion(results['foreign_logits'], foreign_labels)
                
                total_loss_val = (political_loss + social_loss + 
                                economic_loss + foreign_loss)
                
                # calculate pred
                political_pred = torch.argmax(results['political_logits'], dim=1)
                social_pred = torch.argmax(results['social_logits'], dim=1)
                economic_pred = torch.argmax(results['economic_logits'], dim=1)
                foreign_pred = torch.argmax(results['foreign_logits'], dim=1)
                
                political_correct += (political_pred == political_labels).sum().item()
                social_correct += (social_pred == social_labels).sum().item()
                economic_correct += (economic_pred == economic_labels).sum().item()
                foreign_correct += (foreign_pred == foreign_labels).sum().item()
                
                total_samples += len(texts)
                total_loss += total_loss_val.item()
                
                all_political_preds.extend(political_pred.cpu().numpy())
                all_political_labels.extend(political_labels.cpu().numpy())
        
        f1 = f1_score(all_political_labels, all_political_preds, average='weighted')
        
        return {
            'loss': total_loss / len(dataloader),
            'political_accuracy': political_correct / total_samples,
            'social_accuracy': social_correct / total_samples,
            'economic_accuracy': economic_correct / total_samples,
            'foreign_accuracy': foreign_correct / total_samples,
            'political_f1': f1
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
              epochs: int = 10, lr: float = 2e-5, save_path: str = "model_checkpoint.pt"):
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0
        best_val_f1 = 0
        train_history = []
        val_history = []
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_dataloader, optimizer, criterion)
            train_history.append(train_metrics)
            
            val_metrics = self.validate(val_dataloader, criterion)
            val_history.append(val_metrics)
            
            scheduler.step()
            
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Pol_Acc: {train_metrics['political_accuracy']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Pol_Acc: {val_metrics['political_accuracy']:.4f}, "
                       f"F1: {val_metrics['political_f1']:.4f}")
            
            if val_metrics['political_f1'] > best_val_f1:
                best_val_f1 = val_metrics['political_f1']
                best_val_acc = val_metrics['political_accuracy']
                self.model.save_model(save_path)
                logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
        
        history = {
            'train': train_history,
            'val': val_history,
            'best_val_accuracy': best_val_acc,
            'best_val_f1': best_val_f1
        }
        
        with open(save_path.replace('.pt', '_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        return history

def load_data(data_path: str) -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
    df = pd.read_csv(data_path)
    
    texts = df['text'].tolist()
    
    political_label_map = {'left': 0, 'center': 1, 'right': 2}
    political_labels = [political_label_map.get(label.lower(), 1) for label in df['political_lean']]
    
    social_label_map = {'liberal': 0, 'moderate': 1, 'conservative': 2}
    social_labels = [social_label_map.get(label.lower(), 1) for label in df.get('social_lean', ['moderate'] * len(texts))]
    
    economic_label_map = {'left': 0, 'center': 1, 'right': 2}
    economic_labels = [economic_label_map.get(label.lower(), 1) for label in df.get('economic_lean', ['center'] * len(texts))]
    
    foreign_label_map = {'isolationist': 0, 'neutral': 1, 'interventionist': 2}
    foreign_labels = [foreign_label_map.get(label.lower(), 1) for label in df.get('foreign_lean', ['neutral'] * len(texts))]
    
    return texts, political_labels, social_labels, economic_labels, foreign_labels

def create_sample_data():
    sample_data = {
        'text': [
            "We need universal healthcare for all Americans",
            "Lower taxes will stimulate economic growth",
            "Climate change requires immediate government action",
            "We should secure our borders and enforce immigration laws",
            "Education funding should be increased significantly",
            "Free market solutions are best for economic problems",
            "We need stricter gun control laws",
            "Traditional values are important for society",
            "Income inequality is a major problem",
            "Military strength ensures national security"
        ],
        'political_lean': ['left', 'right', 'left', 'right', 'left', 'right', 'left', 'right', 'left', 'right'],
        'social_lean': ['liberal', 'conservative', 'liberal', 'conservative', 'liberal', 'conservative', 'liberal', 'conservative', 'liberal', 'conservative'],
        'economic_lean': ['left', 'right', 'center', 'right', 'left', 'right', 'center', 'right', 'left', 'center'],
        'foreign_lean': ['neutral', 'interventionist', 'neutral', 'interventionist', 'neutral', 'neutral', 'neutral', 'interventionist', 'neutral', 'interventionist']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/sample_training_data.csv', index=False)
    return df

def main():
    parser = argparse.ArgumentParser(description='Train Political Embedding Model')
    parser.add_argument('--data_path', type=str, default='data/sample_training_data.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='data/trained_model.pt',
                       help='Path to save trained model')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample training data')
    
    args = parser.parse_args()
    
    if args.create_sample:
        logger.info("Creating sample training data...")
        create_sample_data()
    
    if not os.path.exists(args.data_path):
        logger.info("Training data not found. Creating sample data...")
        create_sample_data()
    
    logger.info(f"Loading data from {args.data_path}")
    texts, political_labels, social_labels, economic_labels, foreign_labels = load_data(args.data_path)
    
    train_texts, val_texts, train_pol, val_pol, train_soc, val_soc, train_econ, val_econ, train_for, val_for = train_test_split(
        texts, political_labels, social_labels, economic_labels, foreign_labels,
        test_size=0.2, random_state=42, stratify=political_labels
    )
    
    train_dataset = PoliticalTextDataset(train_texts, train_pol, train_soc, train_econ, train_for)
    val_dataset = PoliticalTextDataset(val_texts, val_pol, val_soc, val_econ, val_for)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = IdeologyEmbeddingModel()
    trainer = PoliticalTrainer(model, device)
    
    logger.info("Starting training...")
    history = trainer.train(
        train_dataloader, val_dataloader,
        epochs=args.epochs, lr=args.lr, save_path=args.save_path
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {history['best_val_accuracy']:.4f}")
    logger.info(f"Best validation F1: {history['best_val_f1']:.4f}")

if __name__ == "__main__":
    main() 