// <auto-generated />
#pragma warning disable 1570, 1591

using System;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;

namespace Models
{
	/// <summary>
	/// Generated algorithm for performing inference.
	/// </summary>
	/// <remarks>
	/// If you wish to use this class directly, you must perform the following steps:
	/// 1) Create an instance of the class.
	/// 2) Set the value of any externally-set fields e.g. data, priors.
	/// 3) Call the Execute(numberOfIterations) method.
	/// 4) Use the XXXMarginal() methods to retrieve posterior marginals for different variables.
	/// 
	/// Generated by Infer.NET 0.3.2102.1701 at 18:28 on 2 марта 2021 г..
	/// </remarks>
	public partial class Model0_EP : IGeneratedAlgorithm
	{
		#region Fields
		/// <summary>True if Changed_vy has executed. Set this to false to force re-execution of Changed_vy</summary>
		public bool Changed_vy_isDone;
		/// <summary>Field backing the NumberOfIterationsDone property</summary>
		private int numberOfIterationsDone;
		/// <summary>Field backing the vy property</summary>
		private double Vy;
		/// <summary>Message to marginal of 'vy'</summary>
		public Gaussian vy_marginal_F;
		/// <summary>Message to marginal of 'vyMean'</summary>
		public Gaussian vyMean_marginal_F;
		/// <summary>Message to marginal of 'vySigma'</summary>
		public Gamma vySigma_marginal_F;
		#endregion

		#region Properties
		/// <summary>The number of iterations done from the initial state</summary>
		public int NumberOfIterationsDone
		{
			get {
				return this.numberOfIterationsDone;
			}
		}

		/// <summary>The externally-specified value of 'vy'</summary>
		public double vy
		{
			get {
				return this.Vy;
			}
			set {
				if (this.Vy!=value) {
					this.Vy = value;
					this.numberOfIterationsDone = 0;
					this.Changed_vy_isDone = false;
				}
			}
		}

		#endregion

		#region Methods
		/// <summary>Computations that depend on the observed value of vy</summary>
		private void Changed_vy()
		{
			if (this.Changed_vy_isDone) {
				return ;
			}
			Gaussian vyMean_F = default(Gaussian);
			this.vyMean_marginal_F = Gaussian.Uniform();
			Gaussian vyMean_use_B = default(Gaussian);
			// Message to 'vyMean' from GaussianFromMeanAndVariance factor
			vyMean_F = GaussianFromMeanAndVarianceOp.SampleAverageConditional(0.0, 100.0);
			Gamma vySigma_F = default(Gamma);
			// Message to 'vySigma' from Sample factor
			vySigma_F = GammaFromShapeAndScaleOp.SampleAverageConditional(1.0, 1.0);
			Gamma vySigma_use_B = default(Gamma);
			// Message to 'vySigma_use' from Gaussian factor
			vySigma_use_B = GaussianOp.PrecisionAverageConditional(Gaussian.PointMass(this.Vy), vyMean_F, vySigma_F);
			// Message to 'vyMean_use' from Gaussian factor
			vyMean_use_B = GaussianOp.MeanAverageConditional(this.Vy, vyMean_F, vySigma_F, vySigma_use_B);
			// Message to 'vyMean_marginal' from Variable factor
			this.vyMean_marginal_F = VariableOp.MarginalAverageConditional<Gaussian>(vyMean_use_B, vyMean_F, this.vyMean_marginal_F);
			this.vySigma_marginal_F = Gamma.Uniform();
			// Message to 'vySigma_marginal' from Variable factor
			this.vySigma_marginal_F = VariableOp.MarginalAverageConditional<Gamma>(vySigma_use_B, vySigma_F, this.vySigma_marginal_F);
			this.vy_marginal_F = Gaussian.Uniform();
			// Message to 'vy_marginal' from DerivedVariable factor
			this.vy_marginal_F = DerivedVariableOp.MarginalAverageConditional<Gaussian,double>(this.Vy, this.vy_marginal_F);
			this.Changed_vy_isDone = true;
		}

		/// <summary>Update all marginals, by iterating message passing the given number of times</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		/// <param name="initialise">If true, messages that initialise loops are reset when observed values change</param>
		private void Execute(int numberOfIterations, bool initialise)
		{
			this.Changed_vy();
			this.numberOfIterationsDone = numberOfIterations;
		}

		/// <summary>Update all marginals, by iterating message-passing the given number of times</summary>
		/// <param name="numberOfIterations">The total number of iterations that should be executed for the current set of observed values.  If this is more than the number already done, only the extra iterations are done.  If this is less than the number already done, message-passing is restarted from the beginning.  Changing the observed values resets the iteration count to 0.</param>
		public void Execute(int numberOfIterations)
		{
			this.Execute(numberOfIterations, true);
		}

		/// <summary>Get the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		public object GetObservedValue(string variableName)
		{
			if (variableName=="vy") {
				return this.vy;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName)
		{
			if (variableName=="vyMean") {
				return this.VyMeanMarginal();
			}
			if (variableName=="vySigma") {
				return this.VySigmaMarginal();
			}
			if (variableName=="vy") {
				return this.VyMarginal();
			}
			throw new ArgumentException("This class was not built to infer "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName));
		}

		/// <summary>Get the query-specific marginal distribution of a variable.</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName, string query)
		{
			if (query=="Marginal") {
				return this.Marginal(variableName);
			}
			throw new ArgumentException(((("This class was not built to infer \'"+variableName)+"\' with query \'")+query)+"\'");
		}

		/// <summary>Get the query-specific marginal distribution of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName, string query)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName, query));
		}

		private void OnProgressChanged(ProgressChangedEventArgs e)
		{
			// Make a temporary copy of the event to avoid a race condition
			// if the last subscriber unsubscribes immediately after the null check and before the event is raised.
			EventHandler<ProgressChangedEventArgs> handler = this.ProgressChanged;
			if (handler!=null) {
				handler(this, e);
			}
		}

		/// <summary>Reset all messages to their initial values.  Sets NumberOfIterationsDone to 0.</summary>
		public void Reset()
		{
			this.Execute(0);
		}

		/// <summary>Set the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		/// <param name="value">Observed value</param>
		public void SetObservedValue(string variableName, object value)
		{
			if (variableName=="vy") {
				this.vy = (double)value;
				return ;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>Update all marginals, by iterating message-passing an additional number of times</summary>
		/// <param name="additionalIterations">The number of iterations that should be executed, starting from the current message state.  Messages are not reset, even if observed values have changed.</param>
		public void Update(int additionalIterations)
		{
			this.Execute(checked(this.numberOfIterationsDone+additionalIterations), false);
		}

		/// <summary>
		/// Returns the marginal distribution for 'vy' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Gaussian VyMarginal()
		{
			return this.vy_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'vyMean' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Gaussian VyMeanMarginal()
		{
			return this.vyMean_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'vySigma' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Gamma VySigmaMarginal()
		{
			return this.vySigma_marginal_F;
		}

		#endregion

		#region Events
		/// <summary>Event that is fired when the progress of inference changes, typically at the end of one iteration of the inference algorithm.</summary>
		public event EventHandler<ProgressChangedEventArgs> ProgressChanged;
		#endregion

	}

}