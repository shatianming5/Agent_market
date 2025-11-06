import React from 'react'

type BaseProps = {
  children: React.ReactNode
  className?: string
  style?: React.CSSProperties
}

export function Card({ children, className = '', style }: BaseProps) {
  return (
    <div className={`ui-card card ${className}`.trim()} style={style}>
      {children}
    </div>
  )
}

type HeaderProps = BaseProps & {
  title?: React.ReactNode
  subtitle?: React.ReactNode
  actions?: React.ReactNode
}

export function CardHeader({ title, subtitle, actions, children, className = '', style }: HeaderProps) {
  return (
    <div className={`ui-card__header ${className}`.trim()} style={style}>
      <div className="ui-card__header-main">
        {title ? <h3 className="ui-card__title">{title}</h3> : null}
        {subtitle ? <p className="ui-card__subtitle">{subtitle}</p> : null}
        {children}
      </div>
      {actions ? <div className="ui-card__actions">{actions}</div> : null}
    </div>
  )
}

export function CardBody({ children, className = '', style }: BaseProps) {
  return (
    <div className={`ui-card__body ${className}`.trim()} style={style}>
      {children}
    </div>
  )
}

export function CardFooter({ children, className = '', style }: BaseProps) {
  return (
    <div className={`ui-card__footer ${className}`.trim()} style={style}>
      {children}
    </div>
  )
}

export function SectionTitle({ children, className = '', style }: BaseProps) {
  return (
    <h2 className={`ui-section__title ${className}`.trim()} style={style}>
      {children}
    </h2>
  )
}

export function SectionDescription({ children, className = '', style }: BaseProps) {
  return (
    <p className={`ui-section__description ${className}`.trim()} style={style}>
      {children}
    </p>
  )
}
