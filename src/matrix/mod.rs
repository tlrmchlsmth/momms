//Public Modules
mod matrix;
mod general_stride;
mod row_panel;
mod column_panel;
mod hierarch;

pub use self::matrix::{Scalar,Mat,ResizableBuffer,RoCM};
pub use self::general_stride::{Matrix};
pub use self::row_panel::{RowPanelMatrix};
pub use self::column_panel::{ColumnPanelMatrix};
pub use self::hierarch::{Hierarch,HierarchyNode};
//Private Modules
mod view;
