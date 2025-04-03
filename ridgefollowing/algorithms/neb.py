import numpy as np

class GNEB:
    def __init__(self, energy_surface, x_start, x_end, 
                 num_images=10, k=1.0, convergence_tol=1e-3, 
                 max_iter=1000, tangent_method='simple', 
                 step_size=0.1, climbing_image=True):
        """
        Initialize the GNEB solver.

        Parameters:
            energy_surface: an instance of EnergySurface (or subclass)
            x_start (np.ndarray): initial state (fixed)
            x_end (np.ndarray): final state (fixed)
            num_images (int): number of images including endpoints.
            k (float): spring constant for the virtual springs.
            convergence_tol (float): tolerance for convergence.
            max_iter (int): maximum number of iterations.
            tangent_method (str): method for computing the tangent ('simple' here).
            step_size (float): step size for the gradient descent updates.
            climbing_image (bool): if True, designate the highest-energy interior image as the climbing image.
        """
        self.energy_surface = energy_surface
        self.x_start = np.array(x_start)
        self.x_end = np.array(x_end)
        self.num_images = num_images
        self.k = k
        self.convergence_tol = convergence_tol
        self.max_iter = max_iter
        self.tangent_method = tangent_method
        self.step_size = step_size
        self.climbing_image_enabled = climbing_image

        # Initialize images by linear interpolation between endpoints.
        self.images = self.initialize_images()
        self.reaction_coordinate = self.compute_reaction_coordinate()

    def initialize_images(self):
        """Linearly interpolate images between x_start and x_end."""
        images = [
            self.x_start + (self.x_end - self.x_start) * (i / (self.num_images - 1))
            for i in range(self.num_images)
        ]
        return images

    def compute_reaction_coordinate(self):
        """
        Compute the reaction coordinate for the current chain of images.
        The reaction coordinate is defined as the cumulative Euclidean distance along the path.
        """
        rc = [0.0]
        for i in range(1, self.num_images):
            dist = np.linalg.norm(self.images[i] - self.images[i - 1])
            rc.append(rc[-1] + dist)
        return np.array(rc)

    def compute_tangent(self, i):
        """
        Compute the tangent vector at image i.

        Uses a simple method: compares the energies of the neighboring images 
        and uses the difference toward the higher-energy neighbor.
        """
        # Retrieve energies of adjacent images.
        E_prev = self.energy_surface.energy(self.images[i - 1])
        E_next = self.energy_surface.energy(self.images[i + 1])
        if E_next > E_prev:
            tangent = self.images[i + 1] - self.images[i]
        else:
            tangent = self.images[i] - self.images[i - 1]
        norm = np.linalg.norm(tangent)
        if norm == 0:
            return np.zeros_like(tangent)
        return tangent / norm

    def compute_forces(self):
        """
        Compute the total force on each image.

        The total force consists of:
          - The physical force: negative of the gradient, projected perpendicular to the tangent.
          - The spring force: computed using the differences in the reaction coordinate between images.
          - For the climbing image (if enabled), the force is modified to push the image uphill along the reaction coordinate.
        """
        forces = [np.zeros_like(self.x_start) for _ in range(self.num_images)]
        
        # Update reaction coordinate for the current positions.
        self.reaction_coordinate = self.compute_reaction_coordinate()
        rc = self.reaction_coordinate

        # Determine the climbing image: highest energy among interior images.
        climbing_index = None
        if self.climbing_image_enabled:
            energies = [self.energy_surface.energy(img) for img in self.images]
            interior_energies = energies[1:-1]
            climbing_index = np.argmax(interior_energies) + 1

        # Loop over interior images (endpoints remain fixed).
        for i in range(1, self.num_images - 1):
            # Physical force: negative gradient.
            grad = self.energy_surface.gradient(self.images[i])
            tangent = self.compute_tangent(i)
            # Remove the component of the gradient parallel to the tangent.
            grad_parallel = np.dot(grad, tangent) * tangent
            grad_perp = grad - grad_parallel
            f_true = -grad_perp

            # Spring force: use the reaction coordinate differences.
            d_forward = rc[i + 1] - rc[i]
            d_backward = rc[i] - rc[i - 1]
            f_spring = self.k * (d_forward - d_backward) * tangent

            # Modify force if this is the climbing image.
            if self.climbing_image_enabled and i == climbing_index:
                # For the climbing image, remove the spring contribution and reverse the parallel component.
                f_total = -grad + 2 * grad_parallel
            else:
                f_total = f_true + f_spring

            forces[i] = f_total

        return forces

    def run(self):
        """Run the GNEB optimization until convergence or maximum iterations are reached."""
        for iteration in range(self.max_iter):
            forces = self.compute_forces()
            # Calculate the maximum force magnitude among the interior images.
            max_force = max(np.linalg.norm(forces[i]) for i in range(1, self.num_images - 1))
            if max_force < self.convergence_tol:
                print(f"Converged after {iteration} iterations (max force = {max_force:.3e}).")
                break

            # Update each interior image using a simple gradient descent step.
            for i in range(1, self.num_images - 1):
                self.images[i] += self.step_size * forces[i]
                
            # Update the reaction coordinate after moving the images.
            self.reaction_coordinate = self.compute_reaction_coordinate()

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: max force = {max_force:.3e}")

    def get_path(self):
        """
        Return the current chain of images along with the reaction coordinate.
        
        Returns:
            tuple: (images, reaction_coordinate) where images is a list of np.ndarrays and 
                   reaction_coordinate is a np.ndarray of cumulative distances.
        """
        return self.images, self.reaction_coordinate
