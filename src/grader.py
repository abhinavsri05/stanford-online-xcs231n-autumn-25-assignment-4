#!/usr/bin/env python3
import inspect
import unittest
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import re
from graderUtil import graded, CourseTestRunner, GradedTestCase
from unittest.mock import MagicMock, patch
from PIL import Image

from autograde_utils import if_text_in_py, text_in_cell, assert_allclose


# import student submission
import submission

#########
# HELPERS #
#########


class MockDavisDataset:
    """
    A mock Davis dataset that uses synthetic data for testing
    """

    def __init__(self, num_frames=5, frame_size=224):
        self.num_frames = num_frames
        self.frame_size = frame_size

        # Create synthetic data
        self.frames = [
            torch.randn(3, frame_size, frame_size) for _ in range(num_frames)
        ]
        self.masks = [
            torch.randint(0, 2, (1, frame_size, frame_size)) for _ in range(num_frames)
        ]

    def get_sample(self, index):
        return self.frames, self.masks


class MockModel(torch.nn.Module):
    """
    A mock model for testing
    """
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x, t, model_kwargs={}):
        return x  # Just return the input for simplicity


class MockLossModel(torch.nn.Module):
    """
    Create a custom mock model that returns a defined value
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, t, model_kwargs={}):
        return (
            torch.ones_like(x) * 0.5
        )  # Return constant value for predictable testing


class MockSampleModel(torch.nn.Module):

    """
    # Create a custom mock model that returns a defined value
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, t, model_kwargs={}):
        return torch.zeros_like(x)  # Return zeros for predictable testing


#########
# TESTS #
#########


class Test_1(GradedTestCase):
    """
    DDPM tests
    """

    def setUp(self):

        torch.manual_seed(231)
        np.random.seed(231)

        # Initialize solution and submission diffusion models
        self.sol_model = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.GaussianDiffusion(
                MockModel(),
                image_size=32,
                timesteps=100,
                objective="pred_noise",
                beta_schedule="sigmoid",
            ),
        )

        self.sub_model = submission.GaussianDiffusion(
            MockModel(),
            image_size=32,
            timesteps=100,
            objective="pred_noise",
            beta_schedule="sigmoid",
        )

        # Initialize UNet models for testing
        self.sol_unet = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.Unet(
                dim=32, channels=3, dim_mults=(1, 2, 4), condition_dim=512
            ),
        )

        self.sub_unet = submission.Unet(
            dim=32, channels=3, dim_mults=(1, 2, 4), condition_dim=512
        )

        # Path to the notebook
        self.notebook_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "submission",
            "DDPM.ipynb",
        )

    @graded()
    def test_0(self):
        """1-0-basic: Q sample relative error"""

        # Create test data
        batch_size = 5
        channels = 3
        image_size = 32

        torch.manual_seed(231)
        x_start = torch.randn(batch_size, channels, image_size, image_size)
        t = torch.randint(0, 100, (batch_size,))
        noise = torch.randn_like(x_start)

        # Get outputs from solution and submission
        sol_out = self.sol_model.q_sample(x_start, t, noise)
        sub_out = self.sub_model.q_sample(x_start, t, noise)

        # Check if they match with relative error below 1e-5
        assert_allclose(
            sol_out,
            sub_out,
            err_msg="q_sample implementation does not match the expected output.",
            rtol=1e-5,
        )

    @graded()
    def test_1(self):
        """1-1-basic: Predict start from noise"""

        # Create test data
        batch_size = 5
        channels = 3
        image_size = 32

        torch.manual_seed(231)
        x_t = torch.randn(batch_size, channels, image_size, image_size)
        t = torch.randint(0, 100, (batch_size,))
        noise = torch.randn_like(x_t)

        # Get outputs from solution and submission
        sol_out = self.sol_model.predict_start_from_noise(x_t, t, noise)
        sub_out = self.sub_model.predict_start_from_noise(x_t, t, noise)

        # Check if they match with relative error below 1e-5
        assert_allclose(
            sol_out,
            sub_out,
            err_msg="predict_start_from_noise implementation does not match the expected output.",
            rtol=1e-5,
        )

    @graded(is_hidden=True)
    def test_2(self):
        """1-2-hidden: Predict noise from start"""

        # Create test data
        batch_size = 5
        channels = 3
        image_size = 32

        torch.manual_seed(231)
        x_t = torch.randn(batch_size, channels, image_size, image_size)
        t = torch.randint(0, 100, (batch_size,))
        x_start = torch.randn_like(x_t)

        # Get outputs from solution and submission
        sol_out = self.sol_model.predict_noise_from_start(x_t, t, x_start)
        sub_out = self.sub_model.predict_noise_from_start(x_t, t, x_start)

        # Check if they match with relative error below 1e-5
        assert_allclose(
            sol_out,
            sub_out,
            err_msg="predict_noise_from_start implementation does not match the expected output.",
            rtol=1e-5,
        )

    @graded(is_hidden=True)
    def test_3(self):
        """1-3-hidden: UNet forward pass"""

        # Set deterministic mode for more consistent results
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Create test data
        batch_size = 2
        channels = 3
        image_size = 32

        # Set seeds
        torch.manual_seed(231)
        np.random.seed(231)

        # We need to set the same parameters for both models to ensure fair comparison
        # Let's copy the weights from solution to submission
        try:
            self.sub_unet.load_state_dict(self.sol_unet.state_dict())
            print("Successfully copied weights from solution to submission UNet")
        except Exception as e:
            print(f"Failed to copy weights: {e}")

        # Set both models to eval mode to disable dropouts
        self.sol_unet.eval()
        self.sub_unet.eval()

        # Set seeds again just before generating inputs
        torch.manual_seed(231)
        x = torch.randn(batch_size, channels, image_size, image_size)
        t = torch.randint(0, 100, (batch_size,))
        text_emb = torch.randn(batch_size, 512)
        model_kwargs = {"text_emb": text_emb}

        # Use torch.no_grad to avoid tracking gradients
        with torch.no_grad():
            # Get outputs from solution and submission
            sol_out = self.sol_unet(x, t, model_kwargs)
            sub_out = self.sub_unet(x, t, model_kwargs)

        # Print shapes and a small sample of the outputs
        # print(f"Solution output shape: {sol_out.shape}")
        # print(f"Submission output shape: {sub_out.shape}")
        # print(f"Solution output sample: {sol_out[0, 0, 0, :5]}")
        # print(f"Submission output sample: {sub_out[0, 0, 0, :5]}")

        # Check the absolute max difference
        abs_diff = torch.abs(sol_out - sub_out)
        max_abs_diff = torch.max(abs_diff).item()
        print(f"Max absolute difference: {max_abs_diff}")

        # Check if they match with a higher tolerance (to account for potential numerical differences)
        assert_allclose(
            sol_out.detach(),
            sub_out.detach(),
            err_msg="UNet forward implementation does not match the expected output.",
            rtol=1e-5,
        )

    @graded(is_hidden=True)
    def test_4(self):
        """1-4-hidden: p_losses function implementation"""

        # Create new models with the mock loss model

        sol_loss_model = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.GaussianDiffusion(
                MockLossModel(),
                image_size=32,
                timesteps=100,
                objective="pred_noise",
                beta_schedule="sigmoid",
            ),
        )

        sub_loss_model = submission.GaussianDiffusion(
            MockLossModel(),
            image_size=32,
            timesteps=100,
            objective="pred_noise",
            beta_schedule="sigmoid",
        )

        # Create test data
        batch_size = 5
        channels = 3
        image_size = 32

        torch.manual_seed(231)
        x_start = torch.randn(batch_size, channels, image_size, image_size)

        # Fix random seed for time step selection
        torch.manual_seed(231)
        sol_loss = sol_loss_model.p_losses(x_start)

        torch.manual_seed(231)
        sub_loss = sub_loss_model.p_losses(x_start)

        # Check if losses match with relative error below 1e-6
        assert_allclose(
            sol_loss.detach().numpy(),
            sub_loss.detach().numpy(),
            err_msg="p_losses implementation does not match the expected output.",
            rtol=1e-6,
        )

    @graded(is_hidden=True)
    def test_5(self):
        """1-5-hidden: p_sample function implementation"""

        # Create new models with the mock sample model
        sol_sample_model = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.GaussianDiffusion(
                MockSampleModel(),
                image_size=32,
                timesteps=100,
                objective="pred_noise",
                beta_schedule="sigmoid",
            ),
        )

        sub_sample_model = submission.GaussianDiffusion(
            MockSampleModel(),
            image_size=32,
            timesteps=100,
            objective="pred_noise",
            beta_schedule="sigmoid",
        )

        # Create test data
        batch_size = 5
        channels = 3
        image_size = 32
        t = 50

        torch.manual_seed(231)
        x_t = torch.randn(batch_size, channels, image_size, image_size)

        # Fix random seed for sampling noise
        torch.manual_seed(231)
        sol_x_tm1 = sol_sample_model.p_sample(x_t, t)

        torch.manual_seed(231)
        sub_x_tm1 = sub_sample_model.p_sample(x_t, t)

        # Check if sampled outputs match with relative error below 1e-6
        assert_allclose(
            sol_x_tm1,
            sub_x_tm1,
            err_msg="p_sample implementation does not match the expected output.",
            rtol=1e-6,
        )

    @graded(is_hidden=True)
    def test_6(self):
        """1-6-hidden: Classifier free guidance"""

        try:
            # Set deterministic mode for more consistent results
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Create test data
            batch_size = 2
            channels = 3
            image_size = 32

            # Print some debug info
            print("Testing classifier-free guidance...")

            # Set seeds
            torch.manual_seed(231)
            np.random.seed(231)

            # Make sure both models have the same weights for a fair comparison
            try:
                self.sub_unet.load_state_dict(self.sol_unet.state_dict())
                print("Successfully copied weights from solution to submission UNet")
            except Exception as e:
                print(f"Failed to copy weights: {e}")

            # Set both models to eval mode to disable dropouts
            self.sol_unet.eval()
            self.sub_unet.eval()

            # Create test inputs with fixed seed
            torch.manual_seed(231)
            x_t = torch.randn(batch_size, channels, image_size, image_size)
            t = torch.tensor([50] * batch_size)
            text_emb = torch.randn(batch_size, 512)
            model_kwargs = {"text_emb": text_emb, "cfg_scale": 2.0}

            # Run the guidance method on both models with no grad
            with torch.no_grad():
                sol_output = self.sol_unet.cfg_forward(x_t, t, model_kwargs.copy())
                sub_output = self.sub_unet.cfg_forward(x_t, t, model_kwargs.copy())

            # # Print shapes and a small sample of the outputs
            # print(f"Solution output shape: {sol_output.shape}")
            # print(f"Submission output shape: {sub_output.shape}")
            # print(f"Solution output sample: {sol_output[0, 0, 0, :5]}")
            # print(f"Submission output sample: {sub_output[0, 0, 0, :5]}")

            # Check the absolute max difference
            abs_diff = torch.abs(sol_output - sub_output)
            max_abs_diff = torch.max(abs_diff).item()
            print(f"Max absolute difference: {max_abs_diff}")

            # Check if the outputs match with a relaxed tolerance
            assert_allclose(
                sol_output.detach(),
                sub_output.detach(),
                err_msg="Classifier-free guidance implementation does not match the expected output.",
                rtol=1e-5,
            )

            # If we got here, the test passed
            print("Classifier-free guidance test passed!")
            self.assertTrue(
                True, "Classifier-free guidance implementation works correctly"
            )

        except Exception as e:
            print(f"Error in classifier-free guidance test: {e}")
            self.fail(
                f"Error testing classifier-free guidance implementation: {str(e)}"
            )


class Test_2(GradedTestCase):
    """
    CLIP tests
    """

    def setUp(self):
        # Set up common resources for tests
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Create mock tensors for text and image features
        self.text_features = torch.randn(1, 512, device=self.device)  # 512-dim features
        self.text_features = self.text_features / self.text_features.norm(
            dim=1, keepdim=True
        )

        self.image_features = torch.randn(
            10, 512, device=self.device
        )  # 10 images, 512-dim features
        self.image_features = self.image_features / self.image_features.norm(
            dim=1, keepdim=True
        )

        # Create mock CLIP model
        self.mock_clip_model = MagicMock()

        # ------------------------------------------------------------------
        # Make the mock outputs depend *deterministically* on the inputs so
        # that the behaviour is robust to different student implementations.
        # ------------------------------------------------------------------
        # --- encode_text ----------------------------------------------------
        def _encode_text_side_effect(token_tensor):
            # token_tensor: shape (N, L). Create deterministic but varied features
            # by using the sum of token ids to seed random generation
            N = token_tensor.shape[0]
            features = []
            for i in range(N):
                # Use sum of tokens for this text as seed for deterministic random generation
                seed = int(token_tensor[i].sum().item()) % (2**31 - 1)
                torch.manual_seed(seed)
                feat = torch.randn(512, device=token_tensor.device)
                features.append(feat)
            feats = torch.stack(features)
            return F.normalize(feats, dim=1)

        self.mock_clip_model.encode_text.side_effect = _encode_text_side_effect

        # --- encode_image ---------------------------------------------------
        def _encode_image_side_effect(img_tensor):
            # img_tensor: shape (N, 3, H, W). Create deterministic but varied features
            # by using image content to seed random generation
            N = img_tensor.shape[0]
            features = []
            for i in range(N):
                # Use mean of image pixels as seed for deterministic random generation
                seed = int((img_tensor[i].mean() * 1000000).item()) % (2**31 - 1)
                torch.manual_seed(seed)
                feat = torch.randn(512, device=img_tensor.device)
                features.append(feat)
            feats = torch.stack(features)
            return F.normalize(feats, dim=1)

        self.mock_clip_model.encode_image.side_effect = _encode_image_side_effect

        # --- forward ( __call__ ) ------------------------------------------
        def _forward_side_effect(img_tensor, text_tokens):
            img_feat = _encode_image_side_effect(img_tensor)  # (N,512)
            text_feat = _encode_text_side_effect(text_tokens)  # (M,512)
            logits_per_image = img_feat @ text_feat.T  # (N,M)
            logits_per_text = text_feat @ img_feat.T  # (M,N)
            return logits_per_image, logits_per_text

        self.mock_clip_model.side_effect = _forward_side_effect

        # Set up mock preprocessor that returns different tensors based on input
        def _preprocess_side_effect(pil_image):
            # Convert PIL image back to numpy to get a deterministic hash
            img_array = np.array(pil_image)
            # Use image content hash as seed for deterministic random generation
            seed = int(np.sum(img_array).item()) % (2**31 - 1)
            torch.manual_seed(seed)
            return torch.randn(3, 224, 224)

        self.mock_preprocess = MagicMock(side_effect=_preprocess_side_effect)

        # Create mock images - use PIL images or numpy arrays compatible with PIL
        self.mock_images = []
        np.random.seed(42)  # Set fixed seed for deterministic mock images
        for i in range(10):
            # Create valid PIL-compatible numpy array (HWC format, uint8 type)
            # Use different patterns for each image to ensure variety
            img_array = np.random.randint(
                i * 20, (i + 1) * 20 + 50, (224, 224, 3), dtype=np.uint8
            )
            self.mock_images.append(img_array)

        self.class_texts = ["class1", "class2", "class3", "class4", "class5"]

        # Mock DINO model
        self.mock_dino_model = MagicMock()
        # Set up mock DINO output
        self.mock_dino_output = [
            torch.randn(1, 196, 384) for _ in range(5)
        ]  # Assuming 14x14 patches with 384 features
        self.mock_dino_model.get_intermediate_layers.return_value = [
            {"x": patch} for patch in self.mock_dino_output
        ]

        # Mock dataset frames and masks
        self.mock_frames = [torch.randn(3, 224, 224) for _ in range(5)]
        self.mock_masks = [torch.randint(0, 2, (1, 224, 224)) for _ in range(5)]

        # Create a full mock dataset
        self.mock_davis_dataset = MockDavisDataset()

    @graded()
    def test_0(self):
        """2-0-basic: similarity no loop"""
        # Create inputs
        text_features = torch.randn(5, 512, device=self.device)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        image_features = torch.randn(10, 512, device=self.device)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Get student output
        student_similarity = submission.get_similarity_no_loop(
            text_features, image_features
        )

        # Get reference output
        solution_similarity = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.get_similarity_no_loop(
                text_features, image_features
            ),
        )

        # Compare outputs
        self.assertEqual(
            student_similarity.shape,
            solution_similarity.shape,
            "Student solution has incorrect output shape",
        )
        self.assertTrue(
            torch.allclose(student_similarity, solution_similarity, atol=1e-5),
            "Student solution produces different similarity values than reference",
        )

    @graded()
    def test_1(self):
        """2-1-basic: zero shot classification"""

        with patch('torch.no_grad'):
            # Call student function
            student_result = submission.clip_zero_shot_classifier(
                self.mock_clip_model, 
                self.mock_preprocess, 
                self.mock_images,
                self.class_texts, 
                self.device
            )

            # Call solution function
            solution_result = self.run_with_solution_if_possible(
                submission,
                lambda sub_or_sol: sub_or_sol.clip_zero_shot_classifier(
                    self.mock_clip_model,
                    self.mock_preprocess,
                    self.mock_images,
                    self.class_texts,
                    self.device,
                ),
            )

            # Both should return a list of predicted class labels
            self.assertEqual(type(student_result), type(solution_result),
                          "Student solution has different return type than reference")

            # Check that they have the same length (one prediction per image)
            self.assertEqual(len(student_result), len(solution_result),
                          "Student solution returned different number of predictions")

            # Check that the predicted classes match
            for i, (student_pred, solution_pred) in enumerate(zip(student_result, solution_result)):
                self.assertEqual(student_pred, solution_pred,
                              f"Prediction {i} does not match reference solution")

    @graded(is_hidden=True)
    def test_2(self):
        """2-2-hidden: CLIP image retriever"""

        with patch("torch.no_grad"):
            # Set up the same inputs for both implementations
            mock_clip_model = self.mock_clip_model
            mock_preprocess = self.mock_preprocess
            mock_images = self.mock_images

            # Create student and solution retrievers
            student_retriever = submission.CLIPImageRetriever(
                mock_clip_model, mock_preprocess, mock_images, self.device
            )

            solution_retriever = self.run_with_solution_if_possible(
                submission,
                lambda sub_or_sol: sub_or_sol.CLIPImageRetriever(
                    mock_clip_model, mock_preprocess, mock_images, self.device
                ),
            )

            # Test with a simple query
            query = "test query"
            k = 2

            # Get results from both implementations
            student_result = student_retriever.retrieve(query, k)
            solution_result = solution_retriever.retrieve(query, k)

            # Both should return something - check basic structure
            self.assertIsNotNone(student_result)
            self.assertIsNotNone(solution_result)

            # Extract indices from different possible return formats
            if isinstance(student_result, tuple) and len(student_result) > 0:
                student_indices = student_result[0]
            else:
                student_indices = student_result

            if isinstance(solution_result, tuple) and len(solution_result) > 0:
                solution_indices = solution_result[0]
            else:
                solution_indices = solution_result

            # Convert to lists if they're tensors
            if hasattr(student_indices, "tolist"):
                student_indices = student_indices.tolist()
            if hasattr(solution_indices, "tolist"):
                solution_indices = solution_indices.tolist()

            # Check length
            self.assertEqual(
                len(student_indices),
                len(solution_indices),
                "Student solution returns different number of indices than reference",
            )

            # Since both implementations use the same similarity calculation,
            # they should return the same indices (may need to sort for comparison)
            student_indices_sorted = sorted(student_indices)
            solution_indices_sorted = sorted(solution_indices)

            self.assertEqual(
                student_indices_sorted,
                solution_indices_sorted,
                "Student solution returns different indices than reference",
            )

    @graded(is_hidden=True)
    def test_3(self):
        """2-3-hidden: IOU accuracy"""

        # Path to student's notebook
        notebook_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "submission",
            "CLIP_DINO.ipynb",
        )

        try:
            # Get the output from the tagged cell
            block_text = text_in_cell(notebook_path, "iou_accuracy")

            # Extract IoU values
            first_iou = None
            last_iou = None

            for line in block_text:
                first_match = re.search(
                    r"Mean IoU on first test frames: ([\d\.]+)", line
                )
                last_match = re.search(r"Mean IoU on last test frames: ([\d\.]+)", line)

                if first_match:
                    first_iou = float(first_match.group(1))
                if last_match:
                    last_iou = float(last_match.group(1))

            # Check if we found both values
            if first_iou is None or last_iou is None:
                self.fail("Could not extract IoU values")

            # Check against requirements
            self.assertGreaterEqual(
                first_iou,
                0.45,
                f"Mean IoU on first test frames ({first_iou:.3f}) is below required threshold (0.45)",
            )
            self.assertGreaterEqual(
                last_iou,
                0.50,
                f"Mean IoU on last test frames ({last_iou:.3f}) is below required threshold (0.50)",
            )

        except Exception as e:
            self.fail(f"Error testing IoU accuracy: {str(e)}")

    @graded(is_hidden=True)
    def test_4(self):
        """2-4-hidden: All frames IOU"""

        # Path to student's notebook
        notebook_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "submission",
            "CLIP_DINO.ipynb",
        )

        try:
            # Get the output from the tagged cell
            block_text = text_in_cell(notebook_path, "all_frames_iou")

            # Extract IoU value
            first_frames_iou = None
            for line in block_text:
                match = re.search(r"Mean IoU on all frames: ([\d\.]+)", line)
                if match:
                    first_frames_iou = float(match.group(1))
                    break

            # Check if we found the value
            if first_frames_iou is None:
                self.fail("Could not find Mean IoU value")

            # Check against higher requirement
            self.assertGreaterEqual(
                first_frames_iou,
                0.55,
                f"Mean IoU on all frames ({first_frames_iou:.3f}) is below the higher threshold (0.55)",
            )

        except Exception as e:
            self.fail(f"Error testing first frames IoU: {str(e)}")


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)

if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
