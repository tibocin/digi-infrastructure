"""
Filepath: pcs/src/pcs/utils/qdrant_collections.py
Purpose: Utility functions for downstream applications to access Qdrant collection configuration
Related Components: Qdrant configuration, collection namespaces, multi-tenant applications
Tags: qdrant, collections, utilities, multi-tenant, configuration
"""

from typing import Dict, List, Optional
from pcs.core.config import get_settings


def get_app_collection_name(app_name: str) -> str:
    """
    Get the collection name for a specific application.
    
    Args:
        app_name: Name of the application (e.g., "digi_core", "lernmi", "beep_boop")
        
    Returns:
        Collection name for the application
        
    Examples:
        >>> get_app_collection_name("digi_core")
        'digi_core_knowledge'
        >>> get_app_collection_name("lernmi")
        'lernmi_knowledge'
    """
    settings = get_settings()
    return settings.qdrant.get_collection_name(app_name)


def get_all_collection_names() -> Dict[str, str]:
    """
    Get all collection namespaces as a dictionary.
    
    Returns:
        Dictionary mapping application names to collection names
        
    Example:
        >>> get_all_collection_names()
        {
            'digi_core': 'digi_core_knowledge',
            'lernmi': 'lernmi_knowledge',
            'beep_boop': 'beep_boop_knowledge',
            'pcs': 'pcs_knowledge',
            'stackr': 'stackr_knowledge',
            'satsflow': 'satsflow_knowledge',
            'bitscrow': 'bitscrow_knowledge',
            'devao': 'devao_knowledge',
            'revao': 'revao_knowledge',
            'tibocin_xyz': 'tibocin_xyz_knowledge',
            'lumi_adventures': 'lumi_adventures_knowledge'
        }
    """
    settings = get_settings()
    return settings.qdrant.collection_namespaces


def get_qdrant_config() -> Dict[str, any]:
    """
    Get complete Qdrant configuration for downstream applications.
    
    Returns:
        Dictionary containing Qdrant connection and collection settings
        
    Example:
        >>> config = get_qdrant_config()
        >>> config['host']
        'localhost'
        >>> config['collections']['digi_core']
        'digi_core_knowledge'
    """
    settings = get_settings()
    qdrant = settings.qdrant
    
    return {
        "host": qdrant.host,
        "port": qdrant.port,
        "grpc_port": qdrant.grpc_port,
        "api_key": qdrant.api_key,
        "prefer_grpc": qdrant.prefer_grpc,
        "use_https": qdrant.use_https,
        "default_vector_size": qdrant.default_vector_size,
        "default_distance_metric": qdrant.default_distance_metric,
        "enable_quantization": qdrant.enable_quantization,
        "quantization_type": qdrant.quantization_type,
        "enforce_tenant_isolation": qdrant.enforce_tenant_isolation,
        "collections": qdrant.collection_namespaces
    }


def validate_app_collection(app_name: str) -> bool:
    """
    Validate that a collection exists for the given application.
    
    Args:
        app_name: Name of the application to validate
        
    Returns:
        True if collection is configured, False otherwise
        
    Example:
        >>> validate_app_collection("digi_core")
        True
        >>> validate_app_collection("unknown_app")
        False
    """
    settings = get_settings()
    return app_name.lower().replace("-", "_").replace(" ", "_") in settings.qdrant.collection_namespaces


def get_supported_apps() -> List[str]:
    """
    Get list of all supported application names.
    
    Returns:
        List of application names that have collection configurations
        
    Example:
        >>> get_supported_apps()
        ['digi_core', 'lernmi', 'beep_boop', 'pcs', 'stackr', 'satsflow', 
         'bitscrow', 'devao', 'revao', 'tibocin_xyz', 'lumi_adventures']
    """
    settings = get_settings()
    return list(settings.qdrant.collection_namespaces.keys())


# Convenience functions for common applications
def get_digi_core_collection() -> str:
    """Get collection name for Digi-core application."""
    return get_app_collection_name("digi_core")


def get_lernmi_collection() -> str:
    """Get collection name for Lernmi application."""
    return get_app_collection_name("lernmi")


def get_beep_boop_collection() -> str:
    """Get collection name for Beep-boop application."""
    return get_app_collection_name("beep_boop")


def get_pcs_collection() -> str:
    """Get collection name for PCS application."""
    return get_app_collection_name("pcs")


def get_stackr_collection() -> str:
    """Get collection name for Stackr application."""
    return get_app_collection_name("stackr")


def get_satsflow_collection() -> str:
    """Get collection name for Satsflow application."""
    return get_app_collection_name("satsflow")


def get_bitscrow_collection() -> str:
    """Get collection name for Bitscrow application."""
    return get_app_collection_name("bitscrow")


def get_devao_collection() -> str:
    """Get collection name for Devao application."""
    return get_app_collection_name("devao")


def get_revao_collection() -> str:
    """Get collection name for Revao application."""
    return get_app_collection_name("revao")


def get_tibocin_xyz_collection() -> str:
    """Get collection name for Tibocin XYZ application."""
    return get_app_collection_name("tibocin_xyz")


def get_lumi_adventures_collection() -> str:
    """Get collection name for Lumi Adventures application."""
    return get_app_collection_name("lumi_adventures")
