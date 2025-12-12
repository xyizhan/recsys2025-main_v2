import pandas as pd
import torch
import numpy as np
from pathlib import Path

class HandcraftedFeatureLoader:
    def __init__(self, feature_dir="./handcrafted_features"):
        self.feature_dir = Path(feature_dir)
        self.user_feat_map = None
        self.product_feat_map = None
        
    def load_features(self):
        """加载所有手工特征"""
        user_feat_path = self.feature_dir / "user_features.parquet"
        product_feat_path = self.feature_dir / "product_features.parquet"
        
        user_features = pd.read_parquet(user_feat_path)
        product_features = pd.read_parquet(product_feat_path)
        
        # 创建用户ID到特征的映射
        self.user_feat_map = {}
        for _, row in user_features.iterrows():
            user_id = row['client_id']
            features = [row[col] for col in user_features.columns if col != 'client_id']
            self.user_feat_map[user_id] = torch.FloatTensor(features)
        
        # 创建商品ID到特征的映射
        self.product_feat_map = {}
        for _, row in product_features.iterrows():
            sku = row['sku']
            features = [row[col] for col in product_features.columns if col != 'sku']
            self.product_feat_map[sku] = torch.FloatTensor(features)
            
    def get_user_feature(self, user_id):
        """获取指定用户的特征"""
        if self.user_feat_map is None:
            self.load_features()
        return self.user_feat_map.get(user_id, torch.zeros(len(self.user_feat_map[next(iter(self.user_feat_map))])))
    
    def get_product_feature(self, sku):
        """获取指定商品的特征"""
        if self.product_feat_map is None:
            self.load_features()
        return self.product_feat_map.get(sku, torch.zeros(len(self.product_feat_map[next(iter(self.product_feat_map))])))
    
    def get_feature_dimensions(self):
        """获取特征维度"""
        if self.user_feat_map is None:
            self.load_features()
        user_dim = len(next(iter(self.user_feat_map.values())))
        product_dim = len(next(iter(self.product_feat_map.values())))
        return user_dim, product_dim