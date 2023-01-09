import numpy as np
import math
import time
import os
import pickle
import torch
from torch.autograd.functional import hessian
from torch.nn.utils import _stateless
import copy
from abc import ABC, abstractmethod
from cubic_subproblem_solver import cubic_subproblem_solver


class Trainer(ABC):
    
    def __init__(self, model, dataset, criterion, model_dir, **kwargs):
        
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.weight_decay = kwargs["weight_decay"] if "weight_decay" in kwargs.keys() else 0.
        
        self.MODEL_DIR = model_dir
        self.iter = 0
        self.prev_params = None # parameters at the previous step
        
        self.metrics = {
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
            "grad_norm": [],
            "step_size": [],
            "params_norm": [],
            "time": [],
            "iter": [],
        }
        
        self.hessian_metrics = {
            "lambda_1": [],
            "lambda_n": [],
            "iter": [],
            "spectrum": [],
            "spectrum_iter": [],
            "hessian": [],
            "hessian_iter": [],
        }
    
    def get_metadata(self):
        return {
            "iter": self.iter,
            "prev_params": self.prev_params,
        }
    
    def update_metadata(self, new_metadata):
        self.iter = new_metadata["iter"]
        self.prev_params = new_metadata["prev_params"]
    
    def save(self):
        if not os.path.isdir(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)
        torch.save(self.model.state_dict(), os.path.join(self.MODEL_DIR, "model"))
        with open(os.path.join(self.MODEL_DIR, "metrics"), "wb") as f:
            pickle.dump(self.metrics, f)
        with open(os.path.join(self.MODEL_DIR, "hessian_metrics"), "wb") as f:
            pickle.dump(self.hessian_metrics, f)
        metadata = self.get_metadata()
        with open(os.path.join(self.MODEL_DIR, "metadata"), "wb") as f:
            pickle.dump(metadata, f)
    
    def load(self):
        self.model.load_state_dict(torch.load(os.path.join(self.MODEL_DIR, "model")))
        with open(os.path.join(self.MODEL_DIR, "metrics"), "rb") as f:
            self.metrics = pickle.load(f)
        with open(os.path.join(self.MODEL_DIR, "hessian_metrics"), "rb") as f:
            self.hessian_metrics = pickle.load(f)
        with open(os.path.join(self.MODEL_DIR, "metadata"), "rb") as f:
            metadata = pickle.load(f)
        self.update_metadata(metadata)
    
    def evaluate(self, X, y):
        with torch.no_grad():
            l2_reg = 0. if (self.weight_decay == 0) else (self.weight_decay/2) * sum([p.data.pow(2).sum() for p in self.model.parameters()]).item()
            preds = self.model(X)
            loss = self.criterion(preds, y) + l2_reg
            if preds.shape[1] > 1:
                # categorical cross entropy loss
                preds = torch.argmax(preds, axis=1)
                acc = torch.mean((preds == y).float())
            else:
                # binary crossentropy loss
                preds = preds.round()
                acc = torch.mean((preds == y).float())
        return loss.item(), acc.item()
    
    def calculate_grad_norm_squared(self, X, y):
        preds = self.model(X)
        loss = self.criterion(preds, y)
        self.model.zero_grad()
        loss.backward()
        grad_norm_squared = sum([(p.grad + self.weight_decay * p.data).pow(2).sum() for p in self.model.parameters()])
        return grad_norm_squared.item()
    
    def calculate_hessian(self, X, y):
        
        names = list(n for n, _ in self.model.named_parameters())                
        hessians = hessian(lambda *params: self.criterion(_stateless.functional_call(self.model, {n: p for n, p in zip(names, params)}, X), y), tuple(self.model.parameters()))
        
        hessians = list(hessians)
        n = len(hessians)
        for i in range(n):
            hessians[i] = list(hessians[i])

        params_numels = [p.data.numel() for p in self.model.parameters()]

        for i in range(n):
            for j in range(n):
                hessians[i][j] = hessians[i][j].reshape(params_numels[i], params_numels[j])
            hessians[i] = torch.cat(hessians[i], axis=1)

        hessians = torch.cat(hessians, axis=0)

        if self.weight_decay > 0:
            hessians += self.weight_decay * torch.eye(hessians.shape[0])

        return hessians
    
    def params_norm(self):
        return math.sqrt(sum([p.data.pow(2).sum().item() for p in self.model.parameters()]))
    
    def params_dist(self, another_model_params):
        return math.sqrt(sum([(p.data - p_another_data).pow(2).sum().item() for p, p_another_data in 
                              zip(self.model.parameters(), another_model_params)]))
    
    def update_metrics(self, training_time):
        train_loss, train_acc = self.evaluate(self.dataset["train_data"], self.dataset["train_targets"])
        test_loss, test_acc = self.evaluate(self.dataset["test_data"], self.dataset["test_targets"])
        
        grad_norm = math.sqrt(self.calculate_grad_norm_squared(self.dataset["train_data"], self.dataset["train_targets"]))
        
        step_size = None if self.iter == 0 else self.params_dist(self.prev_params)
        params_norm = self.params_norm()
        
        self.metrics["train_loss"].append(train_loss)
        self.metrics["test_loss"].append(test_loss)
        self.metrics["train_acc"].append(train_acc)
        self.metrics["test_acc"].append(test_acc)
        self.metrics["grad_norm"].append(grad_norm)
        self.metrics["step_size"].append(step_size)
        self.metrics["params_norm"].append(params_norm)
        self.metrics["time"].append(training_time)
        self.metrics["iter"].append(self.iter)
        
        self.prev_params = None
    
    def update_hessian_metrics(self, save_spectrum_every, save_hessian_every, hessian=None):
        if hessian is None:
            hessian = self.calculate_hessian(self.dataset["train_data"], self.dataset["train_targets"])

        spectrum = torch.linalg.eigvalsh(hessian)
        self.hessian_metrics["lambda_1"].append(spectrum[-1])
        self.hessian_metrics["lambda_n"].append(spectrum[0])
        self.hessian_metrics["iter"].append(self.iter)

        if (save_spectrum_every is not None) and (self.iter % save_spectrum_every == 0):
            self.hessian_metrics["spectrum"].append(spectrum)
            self.hessian_metrics["spectrum_iter"].append(self.iter)
        
        if (save_hessian_every is not None) and (self.iter % save_hessian_every == 0):
            self.hessian_metrics["hessian"].append(hessian)
            self.hessian_metrics["hessian_iter"].append(self.iter)
    
    @abstractmethod
    def print_training_stats(self, iteration):
        pass
    
    def prepare_training(self, eval_every, eval_hessian_every, save_spectrum_every, save_hessian_every):
        # calculate metrics for initial model state
        if eval_every is not None:
            self.update_metrics(training_time=0.0)
        if eval_hessian_every is not None:
            self.update_hessian_metrics(save_spectrum_every, save_hessian_every)
    
    @abstractmethod
    def perform_training_loop(self, max_iters, print_every, eval_every, eval_hessian_every, save_spectrum_every,
                              save_hessian_every, save_every):
        pass
    
    def train(self, max_iters, print_every=None, eval_every=None, eval_hessian_every=None, save_spectrum_every=None,
              save_hessian_every=None, save_every=None):
        message = "every time we want to print metrics, it must be evaluated"
        assert (print_every is None) or ((eval_every is not None) and (print_every % eval_every == 0)), message
        message = "every time we evaluate hessian metrics, we must also evaluate simple metrics"
        assert (eval_hessian_every is None) or ((eval_every is not None) and (eval_hessian_every % eval_every == 0)), message
        message = "every time we save spectrum, it must be evaluated"
        assert (save_spectrum_every is None) or ((eval_hessian_every is not None) and (save_spectrum_every % eval_hessian_every == 0)), message
        message = "every time we save hessian, it must be evaluated"
        assert (save_hessian_every is None) or ((eval_hessian_every is not None) and (save_hessian_every % eval_hessian_every == 0)), message

        if self.iter == 0:
            self.prepare_training(eval_every, eval_hessian_every, save_spectrum_every, save_hessian_every)
        
        if print_every is not None:
            self.print_training_stats(self.iter)
        
        self.perform_training_loop(max_iters, print_every, eval_every, eval_hessian_every, save_spectrum_every,
                                   save_hessian_every, save_every)

        return self.metrics.copy(), self.hessian_metrics.copy()


class AdaptiveGDTrainer(Trainer):
    
    def __init__(self, model, dataset, criterion, model_dir, L_0, L_min, **kwargs):
        
        super().__init__(model, dataset, criterion, model_dir, **kwargs)
        self.L = L_0
        self.L_min = L_min
        self.metrics["L"] = []
        self.numerical_tolerance = kwargs["numerical_tolerance"] if "numerical_tolerance" in kwargs.keys() else 1e-4
        self.min_step_size = kwargs["min_step_size"] if "min_step_size" in kwargs.keys() else 1e-6
    
    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["L"] = self.L
        metadata["L_min"] = self.L_min
        return metadata
    
    def update_metadata(self, new_metadata):
        super().update_metadata(new_metadata)
        self.L = new_metadata["L"]
        self.L_min = new_metadata["L_min"]
    
    def update_metrics(self, training_time):
        super().update_metrics(training_time)
        self.metrics["L"].append(self.L)
    
    def print_training_stats(self, iteration):
        try:
            iter_id = self.metrics["iter"].index(iteration)
            print(f'{self.metrics["iter"][iter_id]:>6,d}: train loss = {self.metrics["train_loss"][iter_id]:>9.6f}, \
test loss = {self.metrics["test_loss"][iter_id]:>9.6f}, L = {self.metrics["L"][iter_id]:>12.8f}, \
time = {self.metrics["time"][iter_id]:>7.2f} sec', end='')
        except:
            pass
        try:
            iter_id = self.hessian_metrics["iter"].index(iteration)
            print(f', lambda_n = {self.hessian_metrics["lambda_n"][iter_id]:>9.6f}', end='')
        except:
            pass
        print()
    
    def calculate_loss(self, another_model, X, y):
        with torch.no_grad():
            l2_reg = 0. if (self.weight_decay == 0) else (self.weight_decay/2) * sum([p.data.pow(2).sum() for p in another_model.parameters()]).item()
            preds = another_model(X)
            loss = self.criterion(preds, y) + l2_reg
        return loss.item()
    
    def set_params(self, another_model, new_params):
        for p, d in zip(another_model.parameters(), new_params):
            p.data = d
    
    def perform_training_loop(self, max_iters, print_every, eval_every, eval_hessian_every, save_spectrum_every,
                              save_hessian_every, save_every):
        total_time = self.metrics["time"][-1]
        start_time = time.perf_counter()
        
        while self.iter < max_iters:
            
            if (eval_every is not None) and ((self.iter+1) % eval_every == 0):
                self.prev_params = [p.data.clone() for p in self.model.parameters()]
            
            outputs = self.model(self.dataset["train_data"])
            self.model.zero_grad()
            loss = self.criterion(outputs, self.dataset["train_targets"])
            loss.backward()
            if self.weight_decay != 0:
                l2_reg = (self.weight_decay/2) * sum([p.data.pow(2).sum() for p in self.model.parameters()])
                loss += l2_reg
                for p in self.model.parameters():
                    p.grad += self.weight_decay * p.data
            grad_norm_squared = sum([p.grad.pow(2).sum().item() for p in self.model.parameters()])

            model_next = copy.deepcopy(self.model)
            new_params = [p.data - 1/self.L * p.grad for p in self.model.parameters()]
            self.set_params(model_next, new_params)
            
            while (self.calculate_loss(model_next, self.dataset["train_data"], self.dataset["train_targets"]) >
                   loss.item() - 1/(2*self.L) * grad_norm_squared + self.numerical_tolerance):
                self.L *= 2
                new_params = [p.data - 1/self.L * p.grad for p in self.model.parameters()]
                self.set_params(model_next, new_params)
            
            if self.params_dist([p_next.data for p_next in model_next.parameters()]) < self.min_step_size:
                print("cannot improve anymore")
                if (save_every is not None):
                    total_time += time.perf_counter() - start_time
                    self.save()
                    start_time = time.perf_counter()
                return
            
            self.set_params(self.model, new_params)
            self.iter += 1
            
            if (eval_every is not None) and (self.iter % eval_every == 0):
                total_time += time.perf_counter() - start_time
                self.update_metrics(total_time)
                if (eval_hessian_every is not None) and (self.iter % eval_hessian_every == 0):
                    self.update_hessian_metrics(save_spectrum_every, save_hessian_every)
                if (print_every is not None) and (self.iter % print_every == 0):
                    self.print_training_stats(self.iter)
                start_time = time.perf_counter()
            
            self.L = max(self.L / 2, self.L_min)
            
            if (save_every is not None) and (self.iter % save_every == 0):
                total_time += time.perf_counter() - start_time
                self.save()
                start_time = time.perf_counter()


class AdaptiveCubicNewtonTrainer(Trainer):
    
    def __init__(self, model, dataset, criterion, model_dir, M_0, M_min, **kwargs):
        
        super().__init__(model, dataset, criterion, model_dir, **kwargs)
        self.M = M_0
        self.M_min = M_min
        self.metrics["M"] = []
        self.numerical_tolerance = kwargs["numerical_tolerance"] if "numerical_tolerance" in kwargs.keys() else 1e-6
        self.min_step_size = kwargs["min_step_size"] if "min_step_size" in kwargs.keys() else 1e-6
        
        # placeholder
        self.hessian = None
    
    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["M"] = self.M
        metadata["M_min"] = self.M_min
        return metadata
    
    def update_metadata(self, new_metadata):
        super().update_metadata(new_metadata)
        self.M = new_metadata["M"]
        self.M_min = new_metadata["M_min"]
        self.hessian = self.calculate_hessian(self.dataset["train_data"], self.dataset["train_targets"])
    
    def update_metrics(self, training_time):
        super().update_metrics(training_time)
        self.metrics["M"].append(self.M)
    
    def update_hessian_metrics(self, save_spectrum_every, save_hessian_every):
        hess = self.hessian.clone()
        super().update_hessian_metrics(save_spectrum_every, save_hessian_every, hess)
    
    def print_training_stats(self, iteration):
        try:
            iter_id = self.metrics["iter"].index(iteration)
            print(f'{self.metrics["iter"][iter_id]:>6,d}: train loss = {self.metrics["train_loss"][iter_id]:>9.6f}, \
test loss = {self.metrics["test_loss"][iter_id]:>9.6f}, M = {self.metrics["M"][iter_id]:>12.8f}, \
time = {self.metrics["time"][iter_id]:>7.2f} sec', end='')
        except:
            pass
        try:
            iter_id = self.hessian_metrics["iter"].index(iteration)
            print(f', lambda_n = {self.hessian_metrics["lambda_n"][iter_id]:>9.6f}', end='')
        except:
            pass
        print()
    
    def calculate_loss(self, another_model, X, y):
        with torch.no_grad():
            l2_reg = 0. if (self.weight_decay == 0) else (self.weight_decay/2) * sum([p.data.pow(2).sum() for p in another_model.parameters()]).item()
            preds = another_model(X)
            loss = self.criterion(preds, y) + l2_reg
        return loss.item()
    
    def calculate_gradient(self, X, y):
        preds = self.model(X)
        loss = self.criterion(preds, y)
        self.model.zero_grad()
        loss.backward()
        gradients = [(p.grad + self.weight_decay * p.data) for p in self.model.parameters()]
        grad = torch.cat([g.flatten() for g in gradients], dim=0)
        return grad
    
    def quadratic_form(self, loss, grad, hess, M, h):
        return loss + torch.dot(grad,h) + 0.5 * torch.dot(hess @ h, h) + M/6 * torch.linalg.norm(h,2)**3
    
    def update_model_params(self, another_model, delta):
        params_numels = [p.data.numel() for p in another_model.parameters()]
        t = list(torch.split(delta, params_numels))
        for i, p in enumerate(another_model.parameters()):
            p.data += t[i].reshape(p.data.shape)
    
    def prepare_training(self, eval_every, eval_hessian_every, save_spectrum_every, save_hessian_every):
        self.hessian = self.calculate_hessian(self.dataset["train_data"], self.dataset["train_targets"])
        super().prepare_training(eval_every, eval_hessian_every, save_spectrum_every, save_hessian_every)
    
    def perform_training_loop(self, max_iters, print_every, eval_every, eval_hessian_every, save_spectrum_every,
                              save_hessian_every, save_every):
        total_time = self.metrics["time"][-1]
        start_time = time.perf_counter()
        while self.iter < max_iters:
            
            if (eval_every is not None) and ((self.iter+1) % eval_every == 0):
                self.prev_params = [p.data.clone() for p in self.model.parameters()]
            
            loss = self.calculate_loss(self.model, self.dataset["train_data"], self.dataset["train_targets"])
            grad = self.calculate_gradient(self.dataset["train_data"], self.dataset["train_targets"])
            hess = self.hessian
            
            h = cubic_subproblem_solver(grad, hess, self.M)
            model_next = copy.deepcopy(self.model)
            self.update_model_params(model_next, h)
            
            while (self.calculate_loss(model_next, self.dataset["train_data"], self.dataset["train_targets"]) >
                   loss - self.M/12 * self.params_dist([p_next.data for p_next in model_next.parameters()])**3 +
                   self.numerical_tolerance):
                self.M *= 2
                h = cubic_subproblem_solver(grad, hess, self.M)
                for p, p_next in zip(self.model.parameters(), model_next.parameters()):
                    p_next.data = p.data.clone()
                self.update_model_params(model_next, h)
            
            if self.params_dist([p_next.data for p_next in model_next.parameters()]) < self.min_step_size:
                print("cannot improve anymore")
                if (save_every is not None):
                    total_time += time.perf_counter() - start_time
                    self.save()
                    start_time = time.perf_counter()
                return
                    
            self.update_model_params(self.model, h)
            self.hessian = self.calculate_hessian(self.dataset["train_data"], self.dataset["train_targets"])
            self.iter += 1
            
            if (eval_every is not None) and (self.iter % eval_every == 0):
                total_time += time.perf_counter() - start_time
                self.update_metrics(total_time)
                if (eval_hessian_every is not None) and (self.iter % eval_hessian_every == 0):
                    self.update_hessian_metrics(save_spectrum_every, save_hessian_every)
                if (print_every is not None) and (self.iter % print_every == 0):
                    self.print_training_stats(self.iter)
                start_time = time.perf_counter()
            
            self.M = max(self.M / 2, self.M_min)
            
            if (save_every is not None) and (self.iter % save_every == 0):
                total_time += time.perf_counter() - start_time
                self.save()
                start_time = time.perf_counter()


class CustomTrainer(Trainer):
    
    def __init__(self, model, dataset, criterion, model_dir, OptimizerClass, optimizer_params, batch_size, **kwargs):
        
        super().__init__(model, dataset, criterion, model_dir, **kwargs)
        if self.weight_decay != 0.:
            optimizer_params["weight_decay"] = self.weight_decay
        self.optimizer = OptimizerClass(self.model.parameters(), **optimizer_params)
        self.batch_start = None
        self.batch_size = batch_size
        self.perm = None
    
    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["batch_start"] = self.batch_start
        metadata["batch_size"] = self.batch_size
        metadata["permutation"] = self.perm
        return metadata
    
    def update_metadata(self, new_metadata):
        super().update_metadata(new_metadata)
        self.batch_start = new_metadata["batch_start"]
        self.batch_size = new_metadata["batch_size"]
        self.perm = new_metadata["permutation"]
    
    def save(self):
        super().save()
        torch.save(self.optimizer.state_dict(), os.path.join(self.MODEL_DIR, "optimizer"))
    
    def load(self):
        super().load()
        self.optimizer.load_state_dict(torch.load(os.path.join(self.MODEL_DIR, "optimizer")))
    
    def print_training_stats(self, iteration):
        try:
            iter_id = self.metrics["iter"].index(iteration)
            print(f'{self.metrics["iter"][iter_id]:>6,d}: train loss = {self.metrics["train_loss"][iter_id]:>9.6f}, \
test loss = {self.metrics["test_loss"][iter_id]:>9.6f}, time = {self.metrics["time"][iter_id]:>7.2f} sec', end='')
        except:
            pass
        try:
            iter_id = self.hessian_metrics["iter"].index(iteration)
            print(f', lambda_n = {self.hessian_metrics["lambda_n"][iter_id]:>9.6f}', end='')
        except:
            pass
        print()
    
    def prepare_training(self, eval_every, eval_hessian_every, save_spectrum_every, save_hessian_every):
        super().prepare_training(eval_every, eval_hessian_every, save_spectrum_every, save_hessian_every)
        assert self.batch_size <= self.dataset["train_data"].shape[0]
        self.batch_start = 0
        self.perm = torch.arange(self.dataset["train_data"].shape[0])
    
    def perform_training_loop(self, max_iters, print_every, eval_every, eval_hessian_every, save_spectrum_every,
                              save_hessian_every, save_every):
        total_time = self.metrics["time"][-1]
        start_time = time.perf_counter()
        while self.iter < max_iters:
            
            if (eval_every is not None) and ((self.iter+1) % eval_every == 0):
                self.prev_params = [p.data.clone() for p in self.model.parameters()]
            
            batch_end = self.batch_start + self.batch_size
            outputs = self.model(self.dataset["train_data"][self.perm[self.batch_start:batch_end]])
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, self.dataset["train_targets"][self.perm[self.batch_start:batch_end]])
            loss.backward()
            self.optimizer.step()
            self.iter += 1
            
            if self.iter % eval_every == 0:
                total_time += time.perf_counter() - start_time
                self.update_metrics(total_time)
                if (eval_hessian_every is not None) and (self.iter % eval_hessian_every == 0):
                    self.update_hessian_metrics(save_spectrum_every, save_hessian_every)
                if (print_every is not None) and (self.iter % print_every == 0):
                    self.print_training_stats(self.iter)
                start_time = time.perf_counter()
            
            # we drop the last batch, if it's not full
            if batch_end + self.batch_size <= self.dataset["train_data"].shape[0]:
                self.batch_start = batch_end
            else:
                self.batch_start = 0
                self.perm = torch.randperm(self.dataset["train_data"].shape[0])
            
            if (save_every is not None) and (self.iter % save_every == 0):
                total_time += time.perf_counter() - start_time
                self.save()
                start_time = time.perf_counter()


def print_training_stats(trainer, print_every=1):
    iters = trainer.metrics["iter"] + trainer.hessian_metrics["iter"]
    iters = sorted(list(set(iters)))
    for i in iters:
        if i % print_every == 0:
            trainer.print_training_stats(i)


def get_metrics(trainer):
    metrics = {}
    for key, value in trainer.metrics.items():
        metrics[key] = np.array(value.copy())            
    return metrics


def get_hessian_metrics(trainer):
    keys = ["iter", "lambda_1", "lambda_n"]
    hessian_metrics = {}
    for key in keys:
        hessian_metrics[key] = np.array(trainer.hessian_metrics[key].copy())
    return hessian_metrics


def print_test_accuracy(trainer_):
    max_test_acc = max(trainer_.metrics["test_acc"])
    last_test_acc = trainer_.metrics["test_acc"][-1]
    print(f"Best test accuracy: {100*max_test_acc:.2f}%")
    print(f"Last test accuracy: {100*last_test_acc:.2f}%")
