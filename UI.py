import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import anndata as ad

from config import *
from ANN import *
from Plotting import *

# --------------------------
# Initialize the session state
# --------------------------

def init_session():
    if "dataset" not in st.session_state:
        st.session_state.dataset = {
            "loaded_name": None,  # Current dataset
            "adata": None,        # Current anndata
            "sub_adata": None,    # Final output anndata
            "error": None         
        }

init_session()

@st.cache_data(ttl="1h", show_spinner=False)
def run_prediction(_adata, cell_types):
    """Cache predicted result, avoid repeat calculation"""
    fig1, sub_adata = prediction_ann(_adata, cell_types)
    fig2, sub_adata = plot_UMAP(sub_adata)
    return fig1, fig2, sub_adata


# --------------------------
# UI component: Data source selection
# --------------------------
def render_data_selector():
    col1, col2 = st.columns(2)
    
    with col1:
        # Public selection
        selected_ds = st.selectbox(
            "Public Dataset",
            options=list(PUBLIC_DATASETS.keys()),
            index=0,
            key="public_ds_selector"
        )
        
        # Show dataset description
        if selected_ds != "Unselected":
            ds_info = PUBLIC_DATASETS[selected_ds]
            with st.expander("📖 Dataset discription"):
                st.caption(ds_info["description"])
    
    with col2:
        # File uploader
        uploaded_file = st.file_uploader(
            "Or upload your own file",
            type=["h5ad"],
            key="file_uploader"
        )
    
    return selected_ds, uploaded_file

# --------------------------
# Data loading
# --------------------------
def load_dataset(dataset_name: str):
    """Loading public dataset to memory"""
    config = PUBLIC_DATASETS[dataset_name]
    
    # Display the loading status
    with st.status("⏳ Data loading...", expanded=True) as status:
        try:
            st.write(f"Loading {dataset_name}...")
            adata = config["loader"](config["path"])
            
            # Update the session status
            st.session_state.dataset = {
                "loaded_name": dataset_name,
                "adata": adata,
                "sub_adata": adata,
                "error": None
            }
            
            status.update(label="✅ Done!", state="complete")
            
        except Exception as e:
            st.session_state.dataset["error"] = str(e)
            status.update(label="❌ Loading failure!!", state="error")
            st.error(f"Error: {str(e)}")

def main():
    st.set_page_config(layout="wide")
    st.title("Single-cell circadian rhythm prediction analysis")

    if 'result_generated' not in st.session_state:
        st.session_state.result_generated = False

    # ==================== Data loading ==================== 
    # init_session
    if "dataset" not in st.session_state:
        init_session()

    # ==================== Upper part：Result plotting ====================
    top_col1, top_col2 = st.columns([1, 1])

    with top_col1:
        # Left: Upload dataset or select public dataset

        # Mutual exclusion disabled state
        file_selected = "file_uploader" in st.session_state and st.session_state.file_uploader is not None
        public_selected = "public_dataset_selector" in st.session_state and st.session_state.public_dataset_selector != "Unselected"

        uploaded_file = st.file_uploader(
            "Upload a scRNA-seq dataset (.h5ad/.adata)",
            type=["h5ad"],
            disabled=public_selected,
            key="file_uploader"
        )
        
        # Public dataset selector
        public_dataset = st.selectbox(
            "Or select a public dataset",
            options=list(PUBLIC_DATASETS.keys()),
            index=0,
            disabled=file_selected,
            key="public_dataset_selector"
        )        

        # Loading public dataset
        if public_dataset != "Unselected":
            # Clear uploaded data
            if uploaded_file is not None:
                st.session_state.file_uploader = None
                st.rerun()
            
            if st.session_state.dataset.get("loaded_name") != public_dataset:
                load_dataset(public_dataset)

        if uploaded_file is not None:
            # Clear public dataset selection
            if public_dataset != "Unselected":
                st.session_state.public_dataset_selector = "Unselected"
                st.rerun()
            
            # Loading uploaded files
            try:
                with st.spinner("Parsing uploaded file..."):
                    adata = ad.read_h5ad(uploaded_file)
                    st.session_state.dataset = {
                        "loaded_name": uploaded_file.name,
                        "adata": adata,
                        "sub_adata": adata,
                        "error": None
                    }
            except Exception as e:
                st.error(f"File parsing failure: {str(e)}")
                st.session_state.dataset["error"] = str(e)

    # ==================== Right side data select ====================
    with top_col2:
        # Disabled at first
        has_data_source = uploaded_file or (public_dataset != "Unselected")
        base_disabled = not has_data_source
        
        # Get cell type
        cell_type_options = []
        if has_data_source:
            if st.session_state.dataset["adata"] is not None:
                try:
                    adata = st.session_state.dataset["adata"]
                    cell_type_options = sorted(adata.obs['cell_type'].dropna().unique().tolist())
                    if not cell_type_options:
                        st.error("No cell types found in column")
                except KeyError:
                    st.error("Required column 'cell_type' missing")
                except Exception as e:
                    st.error(f"Data parsing error: {str(e)}")
            else:
                st.warning("Data loading...")

        # Cell-type selector
        cell_type_col1, cell_type_col2 = st.columns([1, 2])
        with cell_type_col1:
            cell_type = st.selectbox(
                "Select cell types", 
                options=cell_type_options if cell_type_options else ["N/A"],
                disabled=base_disabled or not cell_type_options,
                key="main_cell_type"
            )
        
        with cell_type_col2:
            more_cell_types = st.multiselect(
                "Select more cell types (optional)",
                options=cell_type_options if not base_disabled else [],
                default=None,
                disabled=base_disabled,
                help="Multiselect or not select",
                key="additional_cell_types"
            )
        
        model_col = st.columns([1])[0]
        with model_col:
            selected_model = st.selectbox(
                "Select prediction model",
                options=[
                    "SVM",
                    "ANN"
                ],
                index=1,
                disabled=base_disabled,
                key="model_selector"
            )
        
        # btn_col = st.columns([1])[0]
        button_col1, button_col2 = st.columns([1, 4])
        with button_col1:
        # with btn_col:
            predict_disabled = base_disabled or (not cell_type_options)
            if st.button("Predict", disabled=predict_disabled, type="primary", key="predict"):
                # ==================== Param collection and validate ====================
                # Merge all selected cell type
                selected_types = [cell_type] + more_cell_types if more_cell_types else [cell_type]
                
                # Core param
                predict_params = {
                    "cell_types": selected_types,
                    "model": selected_model,
                    "adata": st.session_state.dataset["adata"] 
                }
                
                # Param validation
                if not predict_params["cell_types"]:
                    st.error("Must select at least one cell type")
                    st.stop()
                
                if st.session_state.dataset["adata"] is None:
                    st.error("Dataset didn't uploaded")
                    st.stop()
                
                # ==================== Run prediction ====================
                # Save params to session_state
                st.session_state.predict_params = predict_params
                st.session_state.show_results = True
                # st.session_state.result_generated = True
                # st.rerun()
        
        # with button_col2:
        #     if st.session_state.get("result_generated", False):
        #         csv_data = st.session_state.dataset["sub_adata"].obs.to_csv().encode()
        #         st.download_button(
        #             "⏬ Download Predictions",
        #             data=csv_data,
        #             file_name="results.csv",
        #             mime="text/csv",
        #             key="download_button"
        #         )
        #     # Download Button (disabled at first)
        #     download_disabled = not st.session_state.get("result_generated", False)
        #     # download_disabled = not st.session_state.result_generated
            
        #     # Generate download data
        #     if not download_disabled:
        #         # Convert adata.obs to csv
        #         csv_data = st.session_state.dataset["sub_adata"].obs.to_csv(index=False).encode()
                
        #         # Create download button
        #         st.download_button(
        #             label="⏬ Download Predictions",
        #             data=csv_data,
        #             file_name="prediction_results.csv",
        #             mime="text/csv",
        #             disabled=download_disabled,
        #             help="Download the CSV file containing the prediction results",
        #             key="download_button"
        #         )
        #     else:
        #         # Show disabled placeholder button status
        #         st.button(
        #             "⏬ Download Predictions (Run prediction first)",
        #             disabled=True,
        #             help="The predictive analysis needs to be run first before the results can be downloaded"
        #         )

    # ==================== Lower part：Result plotting ====================
    if st.session_state.get("show_results", False):
        st.divider()
        result_col1, result_col2 = st.columns(2)

        params = st.session_state.predict_params

        # fig1, fig2, sub_adata = run_prediction(
        #     params["adata"],
        #     params["cell_types"]
        # )

        # st.session_state.dataset = {
        #                 "loaded_name": uploaded_file.name,
        #                 "adata": adata,
        #                 "sub_adata": sub_adata,
        #                 "error": None
        #             }
        
        with result_col1:
            # Plot1 (demo)
            st.subheader("Circadian distribution histogram")

            params = st.session_state.predict_params

            fig1, sub_adata = prediction_ann(
                adata=params["adata"],
                cell_types=params["cell_types"]
            )

            st.pyplot(fig1)

            
        with result_col2:
            # Plot2 (demo)
            st.subheader("Circadian distribution in UMAP")

            params = st.session_state.predict_params

            fig2, sub_adata = plot_UMAP(sub_adata)
            st.session_state.dataset["sub_adata"] = sub_adata
            st.session_state.dataset = {
                        "loaded_name": uploaded_file.name,
                        "adata": adata,
                        "sub_adata": sub_adata,
                        "error": None
                    }

            st.pyplot(fig2)

        st.session_state.result_generated = True
        st.toast("Prediction completed! The result file can be downloaded now", icon="✅")

        with button_col2:
            if st.session_state.get("result_generated", False):
                csv_data = st.session_state.dataset["sub_adata"].obs.to_csv().encode()
                st.download_button(
                    "⏬ Download Predictions",
                    data=csv_data,
                    file_name="results.csv",
                    mime="text/csv",
                    key="download_button"
                )

    # ==================== Dynamic prompt information ====================
    if not (uploaded_file or (public_dataset != "Unselected")):
        st.info("⚠️ Please upload a file first or select a public dataset to enable the analysis function")

if __name__ == "__main__":
    main()